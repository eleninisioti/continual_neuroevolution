"""
Population-Based Training (PBT) with PPO on MuJoCo CheetahRun (CONTINUAL).

Implements PBT (Jaderberg et al., 2017) with continual learning: maintains a
population of PPO agents, periodically exploiting top performers and exploring
hyperparameters. Tasks change by modifying physics (gravity or friction).

Network architecture (matches GA for fair comparison):
- Policy: 4 hidden layers of 32 neurons each with tanh, outputs continuous actions
- Value: 4 hidden layers of 32 neurons each with tanh, outputs scalar

Usage:
    python train_PBT_cheetah_continual.py --env CheetahRun --pop_size 512 --gpus 0
    python train_PBT_cheetah_continual.py --env CheetahRun --pbt_mode hp_only --gpus 0
"""

import argparse
import functools
import os
import sys
import time
import pickle
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def _get_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpus' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

_gpu_arg = _get_gpu_arg()
if _gpu_arg:
    os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_arg
    print(f"Setting CUDA_VISIBLE_DEVICES={_gpu_arg}")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import wandb
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mujoco_playground import registry

# Import PBT exploit/explore from gymnax PBT script
from source.gymnax.train_PBT_gymnax import (
    pbt_exploit_and_explore, stack_population_params,
    stack_population_opt_states, unstack_to_population, compute_diversity_stacked,
)


# ============================================================================
# Tee: duplicate stdout to file + console
# ============================================================================

class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()


# ============================================================================
# Networks for Continuous Control
# ============================================================================

class ContinuousPolicyNetwork(nn.Module):
    """MLP policy for continuous actions. Outputs action means; log_std is a
    separate learnable parameter stored outside the network and passed in."""
    hidden_dims: tuple = (32, 32, 32, 32)
    action_dim: int = 6

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)
        mean = nn.Dense(self.action_dim)(x)
        return mean


class ValueNetwork(nn.Module):
    """MLP value network. Outputs scalar state value."""
    hidden_dims: tuple = (32, 32, 32, 32)

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)
        return nn.Dense(1)(x).squeeze(-1)


# ============================================================================
# Continuous PPO primitives
# ============================================================================

def gaussian_log_prob(mean, log_std, action):
    """Log probability of action under diagonal Gaussian."""
    std = jnp.exp(log_std)
    var = std ** 2
    log_prob = -0.5 * (((action - mean) ** 2) / var + 2 * log_std + jnp.log(2 * jnp.pi))
    return jnp.sum(log_prob, axis=-1)


def gaussian_entropy(log_std):
    """Entropy of diagonal Gaussian."""
    return jnp.sum(0.5 * (1.0 + jnp.log(2 * jnp.pi) + 2 * log_std), axis=-1)


def gaussian_sample(key, mean, log_std):
    """Sample from diagonal Gaussian."""
    std = jnp.exp(log_std)
    noise = random.normal(key, mean.shape)
    return mean + std * noise


def gae_advantages(rewards, values, dones, gamma, gae_lambda):
    """Compute GAE advantages for a single environment trajectory."""
    T = rewards.shape[0]

    def scan_fn(carry, t_rev):
        gae = carry
        t = T - 1 - t_rev
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        return gae, gae

    # Append a bootstrap value of 0 at the end
    values_ext = jnp.concatenate([values, jnp.zeros(1)])
    _, advantages = jax.lax.scan(scan_fn, 0.0, jnp.arange(T))
    return advantages[::-1]


# ============================================================================
# Environment helpers (same as GA/RL cheetah continual)
# ============================================================================

def create_modified_env(env_name, task_mod='gravity', multiplier=1.0):
    """Create environment with modified physics (gravity or friction)."""
    from mujoco import mjx

    env = registry.load(env_name)

    if task_mod == 'gravity':
        gravity_z = -9.81 * multiplier
        env.mj_model.opt.gravity[2] = gravity_z
        print(f"  Gravity set: multiplier={multiplier}, gravity_z={gravity_z:.2f}")
    elif task_mod == 'friction':
        env.mj_model.geom_friction[:] *= multiplier
        print(f"  Friction set: multiplier={multiplier}")
    else:
        raise ValueError(f"Unknown task_mod: {task_mod}. Use 'gravity' or 'friction'.")

    env._mjx_model = mjx.put_model(env.mj_model)
    return env


def sample_task_multipliers(num_tasks, default_mult=1.0, low_mult=0.2, high_mult=5.0):
    """Generate multiplier values cycling through default -> low -> high."""
    multiplier_cycle = [default_mult, low_mult, high_mult]
    return [multiplier_cycle[i % 3] for i in range(num_tasks)]


# ============================================================================
# Hyperparameters
# ============================================================================

DEFAULT_CONFIG = {
    "num_generations": 15000,  # 30 tasks × 500 gens/task, matching GA continual
    "task_interval": 500,      # Change task every 500 generations (= gens_per_task for GA)
    "num_envs": 32,             # Parallel episodes (vmapped); kept small for fast JIT
    "num_steps": 1000,         # Steps per episode; total/gen = 32*1000=32k per rollout call
    "num_epochs": 4,           # PPO epochs per generation
    "num_minibatches": 5,      # Minibatches per epoch
    "gamma": 0.97,
    "learning_rate": 1e-4,
    "ent_coef": 1e-2,
    "episode_length": 1000,
    "policy_hidden_dims": (32, 32, 32, 32),
    "value_hidden_dims": (32, 32, 32, 32),
}


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='PBT + PPO on MuJoCo (Continual)')
    parser.add_argument('--env', type=str, default='CheetahRun',
                        help='Environment name (e.g., CheetahRun, WalkerWalk)')
    parser.add_argument('--num_generations', type=int, default=None,
                        help='Override total generations')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)

    # Continual learning: task settings
    parser.add_argument('--num_tasks', type=int, default=30,
                        help='Number of tasks (default 30 = 10 repetitions of 3 values)')
    parser.add_argument('--task_interval', type=int, default=None,
                        help='Change task every N generations (default: derived from num_tasks)')
    parser.add_argument('--task_mod', type=str, default='friction',
                        choices=['gravity', 'friction'],
                        help='Physics parameter to modify (default: friction)')
    parser.add_argument('--gravity_default_mult', type=float, default=1.0)
    parser.add_argument('--gravity_low_mult', type=float, default=0.2)
    parser.add_argument('--gravity_high_mult', type=float, default=5.0)
    parser.add_argument('--friction_default_mult', type=float, default=1.0)
    parser.add_argument('--friction_low_mult', type=float, default=0.2)
    parser.add_argument('--friction_high_mult', type=float, default=5.0)

    # PBT parameters
    parser.add_argument('--pop_size', type=int, default=1,
                        help='Population size (8 is typical for PBT)')
    parser.add_argument('--pbt_interval', type=int, default=10,
                        help='Run PBT exploit/explore every N generations')
    parser.add_argument('--exploit_fraction', type=float, default=0.2,
                        help='Fraction of population to replace each PBT step')
    parser.add_argument('--perturb_factor', type=float, default=0.2,
                        help='Hyperparameter perturbation range [1-pf, 1+pf]')
    parser.add_argument('--pbt_mode', type=str, default='full',
                        choices=['full', 'hp_only', 'weights_only'],
                        help='PBT mode: full (weights+HP), hp_only, weights_only')

    # PPO hyperparameters (None = use defaults)
    parser.add_argument('--num_envs', type=int, default=None)
    parser.add_argument('--num_steps', type=int, default=None)
    parser.add_argument('--episode_length', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--ent_coef', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--num_minibatches', type=int, default=None)
    parser.add_argument('--init_log_std', type=float, default=-0.5,
                        help='Initial log_std for Gaussian policy (default -0.5)')

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='mujoco_evosax')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Wandb run name. If None, auto-generated.')
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--num_eval_episodes', type=int, default=10)

    return parser.parse_args()


def get_hyperparams(args):
    """Get hyperparameters, using defaults when not specified."""
    cfg = DEFAULT_CONFIG
    return {
        'num_generations': args.num_generations if args.num_generations is not None else cfg['num_generations'],
        'task_interval': args.task_interval if args.task_interval is not None else cfg['task_interval'],
        'num_envs': args.num_envs if args.num_envs is not None else cfg['num_envs'],
        'num_steps': args.num_steps if args.num_steps is not None else cfg['num_steps'],
        'episode_length': args.episode_length if args.episode_length is not None else cfg['episode_length'],
        'learning_rate': args.learning_rate if args.learning_rate is not None else cfg['learning_rate'],
        'gamma': args.gamma if args.gamma is not None else cfg['gamma'],
        'gae_lambda': args.gae_lambda,
        'clip_eps': args.clip_eps,
        'vf_coef': args.vf_coef,
        'ent_coef': args.ent_coef if args.ent_coef is not None else cfg['ent_coef'],
        'num_epochs': args.num_epochs if args.num_epochs is not None else cfg['num_epochs'],
        'num_minibatches': args.num_minibatches if args.num_minibatches is not None else cfg['num_minibatches'],
        'policy_hidden_dims': cfg['policy_hidden_dims'],
        'value_hidden_dims': cfg['value_hidden_dims'],
    }


def main():
    args = parse_args()

    env_name = args.env
    seed = args.seed + args.trial
    trial = args.trial
    pop_size = args.pop_size
    task_mod = args.task_mod

    # Get hyperparameters
    hp = get_hyperparams(args)
    num_generations = hp['num_generations']
    task_interval = hp['task_interval']

    print("=" * 60)
    print(f"PBT + PPO on {env_name} (CONTINUAL, MuJoCo)")
    print("=" * 60)

    # Create initial environment to get dimensions
    if task_mod == 'gravity':
        init_mult = args.gravity_default_mult
    else:
        init_mult = args.friction_default_mult
    env = create_modified_env(env_name, task_mod, init_mult)

    key = random.key(seed)
    key, reset_key = random.split(key)
    state = env.reset(reset_key)
    obs_dim = state.obs.shape[-1]
    action_dim = env.action_size

    print(f"  Environment: {env_name}")
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"  Population size: {pop_size}")
    print(f"  PBT interval: {args.pbt_interval} generations")
    print(f"  Exploit fraction: {args.exploit_fraction}")
    print(f"  Perturb factor: {args.perturb_factor}")
    print(f"  PBT mode: {args.pbt_mode}")
    print(f"  Total generations: {num_generations:,}")
    print(f"  Task interval: {task_interval} generations")
    print(f"  Task mod: {task_mod}")
    print(f"  Base hyperparams: num_envs={hp['num_envs']}, num_steps={hp['num_steps']}, "
          f"lr={hp['learning_rate']}, gamma={hp['gamma']}, ent_coef={hp['ent_coef']}")

    # Sample task multipliers
    num_tasks = args.num_tasks
    if task_mod == 'gravity':
        multiplier_list = sample_task_multipliers(
            num_tasks, args.gravity_default_mult, args.gravity_low_mult, args.gravity_high_mult)
    else:
        multiplier_list = sample_task_multipliers(
            num_tasks, args.friction_default_mult, args.friction_low_mult, args.friction_high_mult)
    print(f"  Multiplier values: {[f'{m:.2f}' for m in multiplier_list]}")

    # Output directory
    if args.output_dir is None:
        output_dir = os.path.join(
            REPO_ROOT, "projects", "mujoco",
            f"pbt_{args.pbt_mode}_{env_name}_continual_{task_mod}",
            f"trial_{trial}"
        )
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging to file
    log_file = os.path.join(output_dir, "train.log")
    sys.stdout = Tee(log_file)

    print(f"  Output: {output_dir}")

    # Create gifs directory
    gifs_dir = os.path.join(output_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)

    # Create network templates
    policy_network = ContinuousPolicyNetwork(
        hidden_dims=hp['policy_hidden_dims'], action_dim=action_dim)
    value_network = ValueNetwork(hidden_dims=hp['value_hidden_dims'])

    # ================================================================
    # Initialize population with shared optimizer
    # ================================================================
    shared_tx = optax.inject_hyperparams(optax.adam)(learning_rate=hp['learning_rate'])
    population_states = []
    population_hyperparams = []
    # Per-agent learnable log_std (not part of network params)
    population_log_stds = []

    for i in range(pop_size):
        key, policy_key, value_key, hp_key = random.split(key, 4)
        dummy_obs = jnp.zeros((1, obs_dim))
        p_params = policy_network.init(policy_key, dummy_obs)
        v_params = value_network.init(value_key, dummy_obs)

        # Perturb initial hyperparameters for diversity
        lr_factor = 1.0 + 0.5 * (2.0 * float(random.uniform(hp_key)) - 1.0)
        key, ent_key = random.split(key)
        ent_factor = 1.0 + 0.5 * (2.0 * float(random.uniform(ent_key)) - 1.0)

        lr_i = float(np.clip(hp['learning_rate'] * lr_factor, 1e-6, 1e-2))
        ent_i = float(np.clip(hp['ent_coef'] * ent_factor, 1e-4, 1.0))

        p_state = TrainState.create(apply_fn=policy_network.apply, params=p_params, tx=shared_tx)
        v_state = TrainState.create(apply_fn=value_network.apply, params=v_params, tx=shared_tx)

        # Set per-agent learning rate
        p_state.opt_state.hyperparams['learning_rate'] = jnp.array(lr_i)
        v_state.opt_state.hyperparams['learning_rate'] = jnp.array(lr_i)

        population_states.append((p_state, v_state))
        population_hyperparams.append({'learning_rate': lr_i, 'ent_coef': ent_i})
        population_log_stds.append(jnp.full((action_dim,), args.init_log_std))

    # Count parameters
    policy_param_count = sum(p.size for p in jax.tree_util.tree_leaves(population_states[0][0].params))
    value_param_count = sum(p.size for p in jax.tree_util.tree_leaves(population_states[0][1].params))
    print(f"  Policy params: {policy_param_count:,}, Value params: {value_param_count:,}")

    # Stack population into GPU arrays
    stacked_pp, stacked_vp = stack_population_params(population_states)
    stacked_p_opt, stacked_v_opt = stack_population_opt_states(population_states)
    stacked_log_stds = jnp.stack(population_log_stds)  # (pop_size, action_dim)

    # Initialize wandb
    config = {
        'env': env_name, 'method': 'pbt', 'mode': 'continual',
        'num_generations': num_generations, 'seed': seed, 'trial': trial,
        'pop_size': pop_size, 'pbt_interval': args.pbt_interval,
        'exploit_fraction': args.exploit_fraction, 'perturb_factor': args.perturb_factor,
        'pbt_mode': args.pbt_mode,
        'task_mod': task_mod, 'num_tasks': num_tasks,
        'task_interval': task_interval, 'task_multipliers': multiplier_list,
        'num_envs': hp['num_envs'], 'num_steps': hp['num_steps'],
        'base_learning_rate': hp['learning_rate'], 'gamma': hp['gamma'],
        'base_ent_coef': hp['ent_coef'],
        'num_epochs': hp['num_epochs'], 'num_minibatches': hp['num_minibatches'],
        'policy_hidden_dims': hp['policy_hidden_dims'],
        'value_hidden_dims': hp['value_hidden_dims'],
        'init_log_std': args.init_log_std,
    }
    run_name = args.run_name if args.run_name else f"pbt_{args.pbt_mode}_{env_name}_continual_{task_mod}_trial{trial}"
    wandb.init(project=args.wandb_project, config=config,
               name=run_name,
               reinit=True)
    wandb.define_metric("generation")
    wandb.define_metric("*", step_metric="generation")

    # Training metrics
    best_reward = -float('inf')
    best_agent_idx = 0
    training_metrics_log = []
    start_time = time.time()

    num_updates = num_generations
    timesteps_per_update = hp['num_envs'] * hp['num_steps']

    # PPO constants for JIT
    _batch_size = hp['num_steps'] * hp['num_envs']
    _minibatch_size = _batch_size // hp['num_minibatches']
    _num_mbs = hp['num_minibatches']
    _num_epochs = hp['num_epochs']
    _gamma = hp['gamma']
    _gae_lambda = hp['gae_lambda']
    _clip_eps = hp['clip_eps']
    _vf_coef = hp['vf_coef']
    _num_envs = hp['num_envs']
    _num_steps = hp['num_steps']
    _max_steps = hp['episode_length']

    # Track per-agent fitness for PBT
    population_fitness = np.full(pop_size, -np.inf)

    current_task = 0
    current_multiplier = multiplier_list[0]

    # ================================================================
    # Build JIT-compiled functions for a given environment
    # ================================================================
    def build_jit_fns(env):
        """Build JIT-compiled rollout, train, and evaluate functions for an env."""

        # ---- collect_rollout for a single agent ----
        def collect_rollout(key, policy_params, value_params, log_std):
            """Collect num_envs independent episode rollouts for PPO training.
            
            MJX envs don't support vmap(env.step), so we run each episode
            independently and vmap over episodes.
            Returns rollout dict with shapes (num_steps, num_envs, ...).
            """

            def run_single_episode(ep_key):
                """Run one episode, collecting transitions for PPO."""
                reset_key, step_key = random.split(ep_key)
                init_state = env.reset(reset_key)

                def step_fn(carry, _):
                    state, rng = carry
                    rng, act_key = random.split(rng)

                    obs = state.obs  # (obs_dim,)
                    mean = policy_network.apply(policy_params, obs)
                    value = value_network.apply(value_params, obs)

                    action = gaussian_sample(act_key, mean, log_std)
                    action = jnp.clip(action, -1.0, 1.0)
                    log_prob = gaussian_log_prob(mean, log_std, action)

                    next_state = env.step(state, action)
                    reward = next_state.reward
                    done = next_state.done

                    # Auto-reset on done
                    rng, reset_rng = random.split(rng)
                    reset_state = env.reset(reset_rng)
                    next_state = jax.tree.map(
                        lambda ns, rs: jnp.where(done, rs, ns),
                        next_state, reset_state
                    )

                    return (next_state, rng), {
                        'obs': obs, 'actions': action, 'log_probs': log_prob,
                        'rewards': reward, 'values': value,
                        'dones': done.astype(jnp.float32),
                    }

                (_, _), ep_rollout = jax.lax.scan(
                    step_fn, (init_state, step_key), None, length=_num_steps)
                return ep_rollout  # each leaf: (num_steps, ...)

            ep_keys = random.split(key, _num_envs)
            # vmap over episodes: each leaf becomes (num_envs, num_steps, ...)
            rollouts = jax.vmap(run_single_episode)(ep_keys)
            # Transpose to (num_steps, num_envs, ...) for consistency with PPO
            rollouts = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), rollouts)
            return rollouts

        # ---- fused rollout + train ----
        def rollout_and_train(key, stacked_pp, stacked_vp, stacked_p_opt, stacked_v_opt,
                              ent_coefs, stacked_log_stds):
            rollout_key, train_key = random.split(key)

            rollout_keys = random.split(rollout_key, pop_size)
            pop_rollouts = jax.vmap(
                lambda k, pp, vp, ls: collect_rollout(k, pp, vp, ls)
            )(rollout_keys, stacked_pp, stacked_vp, stacked_log_stds)

            train_keys = random.split(train_key, pop_size)

            def train_single(key, pp, vp, p_opt, v_opt, rollout, ent_coef, log_std):
                # GAE advantages per environment
                rewards_T = rollout['rewards'].T  # (num_envs, num_steps)
                values_T = rollout['values'].T
                dones_T = rollout['dones'].T
                advantages_T = jax.vmap(
                    lambda r, v, d: gae_advantages(r, v, d, _gamma, _gae_lambda)
                )(rewards_T, values_T, dones_T)
                returns_T = advantages_T + values_T
                advantages = advantages_T.T
                returns = returns_T.T
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Flatten
                flat_obs = rollout['obs'].reshape(_batch_size, -1)
                flat_actions = rollout['actions'].reshape(_batch_size, -1)
                flat_log_probs = rollout['log_probs'].reshape(_batch_size)
                flat_advantages = advantages.reshape(_batch_size)
                flat_returns = returns.reshape(_batch_size)

                def epoch_step(carry, _):
                    pp, vp, p_opt, v_opt, log_std, key = carry
                    key, shuffle_key = random.split(key)
                    perm = random.permutation(shuffle_key, _batch_size)

                    minibatches = {
                        'obs': flat_obs[perm].reshape(_num_mbs, _minibatch_size, -1),
                        'actions': flat_actions[perm].reshape(_num_mbs, _minibatch_size, -1),
                        'log_probs': flat_log_probs[perm].reshape(_num_mbs, _minibatch_size),
                        'advantages': flat_advantages[perm].reshape(_num_mbs, _minibatch_size),
                        'returns': flat_returns[perm].reshape(_num_mbs, _minibatch_size),
                    }

                    def minibatch_step(carry, mb):
                        pp, vp, p_opt, v_opt, log_std = carry

                        def loss_fn(policy_params, value_params, log_std_param):
                            obs = mb['obs']
                            actions = mb['actions']
                            old_log_probs = mb['log_probs']
                            advantages = mb['advantages']
                            returns = mb['returns']

                            means = jax.vmap(
                                policy_network.apply, in_axes=(None, 0)
                            )(policy_params, obs)
                            new_log_probs = gaussian_log_prob(means, log_std_param, actions)
                            ratio = jnp.exp(new_log_probs - old_log_probs)

                            # Clipped surrogate objective
                            pg_loss1 = -advantages * ratio
                            pg_loss2 = -advantages * jnp.clip(
                                ratio, 1.0 - _clip_eps, 1.0 + _clip_eps)
                            pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

                            # Value loss
                            values = jax.vmap(
                                value_network.apply, in_axes=(None, 0)
                            )(value_params, obs)
                            vf_loss = jnp.mean((values - returns) ** 2)

                            # Entropy
                            entropy = jnp.mean(gaussian_entropy(log_std_param))

                            total_loss = pg_loss + _vf_coef * vf_loss - ent_coef * entropy
                            return total_loss

                        (p_grads, v_grads, ls_grads) = jax.grad(
                            loss_fn, argnums=(0, 1, 2))(pp, vp, log_std)

                        p_updates, new_p_opt = shared_tx.update(p_grads, p_opt, pp)
                        pp = optax.apply_updates(pp, p_updates)
                        v_updates, new_v_opt = shared_tx.update(v_grads, v_opt, vp)
                        vp = optax.apply_updates(vp, v_updates)

                        # Update log_std with a simple gradient step (use same lr)
                        lr = new_p_opt.hyperparams['learning_rate']
                        log_std = log_std - lr * ls_grads

                        return (pp, vp, new_p_opt, new_v_opt, log_std), None

                    (pp, vp, p_opt, v_opt, log_std), _ = jax.lax.scan(
                        minibatch_step, (pp, vp, p_opt, v_opt, log_std), minibatches)
                    return (pp, vp, p_opt, v_opt, log_std, key), None

                (pp, vp, p_opt, v_opt, log_std, _), _ = jax.lax.scan(
                    epoch_step, (pp, vp, p_opt, v_opt, log_std, key), None,
                    length=_num_epochs)
                return pp, vp, p_opt, v_opt, log_std

            new_pp, new_vp, new_p_opt, new_v_opt, new_log_stds = jax.vmap(train_single)(
                train_keys, stacked_pp, stacked_vp, stacked_p_opt, stacked_v_opt,
                pop_rollouts, ent_coefs, stacked_log_stds)
            return new_pp, new_vp, new_p_opt, new_v_opt, new_log_stds

        jit_rollout_and_train = jax.jit(rollout_and_train)

        # ---- vmapped evaluation ----
        num_eval_eps = args.num_eval_episodes

        def evaluate_population(keys, stacked_pp, stacked_log_stds):
            def eval_single(key, policy_params, log_std):
                def run_episode(ep_key):
                    reset_key, step_key = random.split(ep_key)
                    state = env.reset(reset_key)

                    def step_fn(carry, _):
                        state, total_reward, done_flag, k = carry
                        k, act_key = random.split(k)
                        mean = policy_network.apply(policy_params, state.obs)
                        action = jnp.clip(mean, -1.0, 1.0)  # deterministic
                        next_state = env.step(state, action)
                        reward = next_state.reward
                        done = next_state.done
                        total_reward = total_reward + reward * (1.0 - done_flag)
                        done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
                        return (next_state, total_reward, done_flag, k), None

                    (_, total_reward, _, _), _ = jax.lax.scan(
                        step_fn, (state, 0.0, 0.0, step_key), None,
                        length=_max_steps)
                    return total_reward

                ep_keys = random.split(key, num_eval_eps)
                rewards = jax.vmap(run_episode)(ep_keys)
                return jnp.mean(rewards)

            return jax.vmap(eval_single)(keys, stacked_pp, stacked_log_stds)

        jit_evaluate_population = jax.jit(evaluate_population)

        return jit_rollout_and_train, jit_evaluate_population

    # ================================================================
    # Helper: save GIFs for best agent at end of task
    # ================================================================
    def save_task_gifs(task_idx, task_label, best_policy_params, best_log_std, env):
        nonlocal key
        try:
            num_gifs = 10
            task_gifs_dir = os.path.join(gifs_dir, f"task_{task_idx:02d}_{task_label}")
            os.makedirs(task_gifs_dir, exist_ok=True)

            jit_reset = jax.jit(env.reset)
            jit_step = jax.jit(env.step)

            for gif_idx in range(num_gifs):
                key, gif_key = random.split(key)
                state = jit_reset(gif_key)
                rollout_states = [state]
                total_reward = 0.0

                for step in range(_max_steps):
                    mean = policy_network.apply(best_policy_params, state.obs)
                    action = jnp.clip(mean, -1.0, 1.0)
                    state = jit_step(state, action)
                    rollout_states.append(state)
                    total_reward += float(state.reward)
                    if state.done:
                        break

                images = env.render(rollout_states[::2], height=240, width=320, camera="side")
                gif_path = os.path.join(
                    task_gifs_dir,
                    f"task{task_idx}_rollout_{gif_idx:02d}_reward{total_reward:.0f}.gif")
                imageio.mimsave(gif_path, images, fps=30, loop=0)

            print(f"  Saved {num_gifs} GIFs for task {task_idx} ({task_label})")
        except Exception as e:
            print(f"  Warning: Failed to save GIFs for task {task_idx}: {e}")

    # ================================================================
    # Cache ent_coefs
    # ================================================================
    ent_coefs = jnp.array([population_hyperparams[i]['ent_coef'] for i in range(pop_size)])

    # ================================================================
    # Main training loop
    # ================================================================
    print(f"\nStarting PBT continual training ({num_updates} generations)...")

    # Build JIT functions for initial environment
    print(f"\n  Task 0 ({task_mod}={current_multiplier:.2f}): building JIT functions...")
    jit_rollout_and_train, jit_evaluate_population = build_jit_fns(env)

    for update in range(num_updates):
        timestep = (update + 1) * timesteps_per_update

        # ================================================================
        # Check for task switch
        # ================================================================
        if update > 0 and update % task_interval == 0:
            # End-of-task evaluation
            key, eval_rng = random.split(key)
            eval_keys = random.split(eval_rng, pop_size)
            task_eval_rewards_jax = jit_evaluate_population(
                eval_keys, stacked_pp, stacked_log_stds)
            task_eval_rewards = np.array(task_eval_rewards_jax)

            task_best_idx = int(np.argmax(task_eval_rewards))
            task_best_reward = task_eval_rewards[task_best_idx]
            task_mean_reward = float(np.mean(task_eval_rewards))
            task_diversity = compute_diversity_stacked(stacked_pp)

            print(f"\n  End-of-task {current_task} evaluation:")
            print(f"    Best: agent {task_best_idx} = {task_best_reward:.2f}")
            print(f"    Mean: {task_mean_reward:.2f} | Diversity: {task_diversity:.4f}")

            wandb.log({
                'generation': update + 1,
                'task_eval/task': current_task,
                'task_eval/best_reward': task_best_reward,
                'task_eval/mean_reward': task_mean_reward,
                'task_eval/best_agent': task_best_idx,
                'task_eval/diversity': task_diversity,
            })

            # Save GIFs for ending task
            task_label = f"{task_mod}_{current_multiplier:.2f}".replace(".", "p")
            best_pp = jax.tree.map(lambda x: x[task_best_idx], stacked_pp)
            best_ls = stacked_log_stds[task_best_idx]
            save_task_gifs(current_task, task_label, best_pp, best_ls, env)

            # Switch to next task
            current_task += 1
            if current_task < num_tasks:
                current_multiplier = multiplier_list[current_task]
                print(f"\n>>> Task {current_task} started at generation {update}")
                print(f"    {task_mod} multiplier: {current_multiplier:.2f}")
                env = create_modified_env(env_name, task_mod, current_multiplier)
                # Rebuild JIT functions for new environment
                jit_rollout_and_train, jit_evaluate_population = build_jit_fns(env)

        # ================================================================
        # Rollout + Train (fused JIT)
        # ================================================================
        key, step_rng = random.split(key)
        stacked_pp, stacked_vp, stacked_p_opt, stacked_v_opt, stacked_log_stds = \
            jit_rollout_and_train(
                step_rng, stacked_pp, stacked_vp, stacked_p_opt, stacked_v_opt,
                ent_coefs, stacked_log_stds)

        # ================================================================
        # PBT exploit/explore step
        # ================================================================
        if (update + 1) % args.pbt_interval == 0:
            key, eval_rng = random.split(key)
            eval_keys = random.split(eval_rng, pop_size)
            pop_rewards = jit_evaluate_population(
                eval_keys, stacked_pp, stacked_log_stds)
            population_fitness = np.array(pop_rewards)

            diversity = compute_diversity_stacked(stacked_pp)
            print(f"\n  PBT step at generation {update+1} (Task {current_task}):")
            print(f"    Best: agent {np.argmax(population_fitness)} = {np.max(population_fitness):.2f}")
            print(f"    Mean: {np.mean(population_fitness):.2f} | Diversity: {diversity:.4f}")

            # Unstack for PBT
            population_states = unstack_to_population(
                stacked_pp, stacked_vp, stacked_p_opt, stacked_v_opt,
                shared_tx, policy_network, value_network, pop_size)

            # Exploit + Explore
            key, pbt_key = random.split(key)
            population_states, population_hyperparams = pbt_exploit_and_explore(
                population_states, population_hyperparams, population_fitness,
                pbt_key,
                exploit_fraction=args.exploit_fraction,
                perturb_factor=args.perturb_factor,
                mode=args.pbt_mode,
                shared_tx=shared_tx,
            )

            # Also copy log_stds from top → bottom during weight exploitation
            if args.pbt_mode in ('full', 'weights_only'):
                num_replace = max(1, int(pop_size * args.exploit_fraction))
                sorted_indices = np.argsort(population_fitness)
                bottom_indices = sorted_indices[:num_replace]
                top_indices = sorted_indices[-num_replace:]
                log_std_list = [stacked_log_stds[i] for i in range(pop_size)]
                for i, bot_idx in enumerate(bottom_indices):
                    key, sel_key = random.split(key)
                    top_idx = top_indices[int(random.randint(sel_key, (), 0, num_replace))]
                    log_std_list[bot_idx] = log_std_list[top_idx]
                stacked_log_stds = jnp.stack(log_std_list)

            # Restack after PBT
            stacked_pp, stacked_vp = stack_population_params(population_states)
            stacked_p_opt, stacked_v_opt = stack_population_opt_states(population_states)

            # Update cached ent_coefs
            ent_coefs = jnp.array([population_hyperparams[i]['ent_coef']
                                   for i in range(pop_size)])

            diversity_after = compute_diversity_stacked(stacked_pp)

            # Log hyperparams
            for i in range(pop_size):
                wandb.log({
                    f'pbt/agent{i}/learning_rate': population_hyperparams[i]['learning_rate'],
                    f'pbt/agent{i}/ent_coef': population_hyperparams[i]['ent_coef'],
                    f'pbt/agent{i}/fitness': population_fitness[i],
                    'generation': update + 1,
                    'task': current_task,
                })
            wandb.log({
                'pbt/diversity_before': diversity,
                'pbt/diversity_after': diversity_after,
                'pbt/fitness_max': float(np.max(population_fitness)),
                'pbt/fitness_mean': float(np.mean(population_fitness)),
                'generation': update + 1,
                'task': current_task,
            })

        # ================================================================
        # Periodic evaluation
        # ================================================================
        if (update + 1) % args.eval_interval == 0 or update == num_updates - 1:
            key, eval_rng = random.split(key)
            eval_keys = random.split(eval_rng, pop_size)
            eval_rewards_jax = jit_evaluate_population(
                eval_keys, stacked_pp, stacked_log_stds)
            eval_rewards = np.array(eval_rewards_jax)

            best_agent_idx = int(np.argmax(eval_rewards))
            pop_best_reward = eval_rewards[best_agent_idx]
            pop_mean_reward = float(np.mean(eval_rewards))

            if pop_best_reward > best_reward:
                best_reward = pop_best_reward

            diversity = compute_diversity_stacked(stacked_pp)
            elapsed = time.time() - start_time

            training_metrics_log.append({
                'timestep': timestep,
                'update': update + 1,
                'task': current_task,
                'multiplier': current_multiplier,
                'best_reward': float(best_reward),
                'pop_best_reward': float(pop_best_reward),
                'pop_mean_reward': float(pop_mean_reward),
                'best_agent': best_agent_idx,
                'diversity': diversity,
                'elapsed_time': elapsed,
            })

            wandb.log({
                'generation': update + 1,
                'timestep': timestep,
                'task': current_task,
                'multiplier': current_multiplier,
                'eval/pop_best_reward': pop_best_reward,
                'eval/pop_mean_reward': pop_mean_reward,
                'eval/best_reward': best_reward,
                'eval/best_agent': best_agent_idx,
                'eval/diversity': diversity,
            })

            print(f"Gen {update+1:6d} | Task {current_task} | "
                  f"{task_mod}={current_multiplier:.2f} | "
                  f"Pop Best: {pop_best_reward:8.2f} (agent {best_agent_idx}) | "
                  f"Pop Mean: {pop_mean_reward:8.2f} | "
                  f"Diversity: {diversity:.4f} | "
                  f"Time: {elapsed:.1f}s", flush=True)

    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s, Best: {best_reward:.2f}")

    # ================================================================
    # Final evaluation of best agent
    # ================================================================
    population_states = unstack_to_population(
        stacked_pp, stacked_vp, stacked_p_opt, stacked_v_opt,
        shared_tx, policy_network, value_network, pop_size)

    print(f"\nFinal evaluation of best agent (agent {best_agent_idx}) on Task {current_task}...")
    best_policy_state, best_value_state = population_states[best_agent_idx]
    best_log_std = stacked_log_stds[best_agent_idx]

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    final_eval_rewards = []
    for eval_trial in range(10):
        key, eval_key = random.split(key)
        state = jit_reset(eval_key)
        trial_reward = 0.0
        for step in range(_max_steps):
            mean = policy_network.apply(best_policy_state.params, state.obs)
            action = jnp.clip(mean, -1.0, 1.0)
            state = jit_step(state, action)
            trial_reward += float(state.reward)
            if bool(state.done):
                break
        final_eval_rewards.append(trial_reward)
        print(f"  Trial {eval_trial + 1}: {trial_reward:.2f}")

    final_mean = float(np.mean(final_eval_rewards))
    final_std = float(np.std(final_eval_rewards))
    final_max = float(np.max(final_eval_rewards))
    final_min = float(np.min(final_eval_rewards))

    print(f"\nFinal evaluation results:")
    print(f"  Mean: {final_mean:.2f} +/- {final_std:.2f}")
    print(f"  Min: {final_min:.2f}, Max: {final_max:.2f}")
    print(f"  Training best: {best_reward:.2f}")

    wandb.log({
        "final_eval_mean": final_mean,
        "final_eval_std": final_std,
        "final_eval_max": final_max,
        "final_eval_min": final_min,
    })

    # Save GIFs for final task
    final_task_label = f"{task_mod}_{current_multiplier:.2f}".replace(".", "p")
    save_task_gifs(current_task, final_task_label,
                   best_policy_state.params, best_log_std, env)

    # Save checkpoint
    ckpt_path = os.path.join(output_dir,
                             f"pbt_{env_name}_continual_{task_mod}_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'policy_params': best_policy_state.params,
            'value_params': best_value_state.params,
            'log_std': np.array(best_log_std),
            'best_reward': best_reward,
            'final_eval_mean': final_mean,
            'final_eval_std': final_std,
            'best_agent_idx': best_agent_idx,
            'population_hyperparams': population_hyperparams,
            'config': config,
        }, f)
    print(f"Saved: {ckpt_path}")

    # Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics_log, f, indent=2)

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
