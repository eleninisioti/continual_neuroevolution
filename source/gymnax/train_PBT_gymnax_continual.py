"""
Population-Based Training (PBT) with PPO on Gymnax environments (CONTINUAL).

Implements PBT (Jaderberg et al., 2017) with continual learning: maintains a
population of PPO agents, periodically exploiting top performers and exploring
hyperparameters. Tasks change every `task_interval` updates via observation noise.

Supports CartPole-v1, Acrobot-v1, MountainCar-v0.

Network architecture (same as standard PPO):
- Policy: 2 hidden layers of 16 neurons each with ReLU
- Value: 3 hidden layers of 128 neurons each with ReLU

Usage:
    python train_PBT_gymnax_continual.py --env CartPole-v1 --pop_size 8 --gpus 0
    python train_PBT_gymnax_continual.py --env Acrobot-v1 --pop_size 16 --gpus 0
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

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import gymnax
import wandb
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import from the continual RL script (has noise-aware collect_rollout and evaluate)
from source.gymnax.train_RL_gymnax_continual import (
    render_cartpole_frame, render_acrobot_frame, render_mountaincar_frame,
    get_render_fn, PolicyNetwork, ValueNetwork,
    categorical_sample, categorical_log_prob, categorical_entropy,
    gae_advantages, collect_rollout, compute_ppo_loss, train_step,
    evaluate, Tee,
)

# Import PBT exploit/explore from non-continual PBT script
from source.gymnax.train_PBT_gymnax import (
    pbt_exploit_and_explore, compute_population_diversity, stack_population_params, 
    stack_population_opt_states, unstack_to_population, compute_diversity_stacked
)


# ============================================================================
# Environment-Specific Hyperparameters (base values for PBT)
# ============================================================================

ENV_CONFIGS = {
    "CartPole-v1": {
        "num_generations": 4000,  # 20 tasks × 200 generations, matching GA
        "task_interval": 200,  # 200 generations per task
        "num_envs": 2048,  # Match RL for fair comparison
        "num_steps": 50,   # Match RL unroll_length
        "num_epochs": 10,
        "num_minibatches": 32,
        "gamma": 0.95,
        "learning_rate": 3e-4,
        "ent_coef": 1e-2,
        "normalize_observations": True,
        "policy_hidden_dims": (16, 16),
        "value_hidden_dims": (128, 128, 128),
        "episode_length": 500,
    },
    "Acrobot-v1": {
        "num_generations": 4000,  # 20 tasks × 200 generations, matching GA
        "task_interval": 200,
        "num_envs": 2048,  # Match RL for fair comparison
        "num_steps": 50,   # Match RL unroll_length
        "num_epochs": 10,
        "num_minibatches": 32,
        "gamma": 0.99,
        "learning_rate": 1e-4,
        "ent_coef": 1e-2,
        "normalize_observations": True,
        "policy_hidden_dims": (16, 16),
        "value_hidden_dims": (128, 128, 128),
        "episode_length": 500,
    },
    "MountainCar-v0": {
        "num_generations": 4000,  # 20 tasks × 200 generations, matching GA
        "task_interval": 200,
        "num_envs": 2048,  # Match RL for fair comparison
        "num_steps": 50,   # Match RL unroll_length
        "num_epochs": 10,
        "num_minibatches": 32,
        "gamma": 0.95,
        "learning_rate": 3e-4,
        "ent_coef": 1e-2,
        "normalize_observations": True,
        "policy_hidden_dims": (16, 16),
        "value_hidden_dims": (128, 128, 128),
        "episode_length": 500,
    },
}


# ============================================================================
# Main Training Loop
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='PBT + PPO on Gymnax (Continual)')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'])
    parser.add_argument('--num_generations', type=int, default=None,
                        help='Override default number of generations for env')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)

    # Continual learning parameters
    parser.add_argument('--task_interval', type=int, default=None,
                        help='Change task every N updates (default: env-specific for 10 tasks)')
    parser.add_argument('--noise_range', type=float, default=1.0,
                        help='Scale for observation noise (task definition)')

    # PBT parameters
    parser.add_argument('--pop_size', type=int, default=512,
                        help='Population size')
    parser.add_argument('--pbt_interval', type=int, default=10,
                        help='Run PBT exploit/explore every N generations')
    parser.add_argument('--exploit_fraction', type=float, default=0.2,
                        help='Fraction of population to replace each PBT step')
    parser.add_argument('--perturb_factor', type=float, default=0.2,
                        help='Hyperparameter perturbation range [1-pf, 1+pf]')
    parser.add_argument('--pbt_mode', type=str, default='weights_only',
                        choices=['full', 'hp_only', 'weights_only'],
                        help='PBT mode: full (weights+HP), hp_only (HP perturbation only), '
                             'weights_only (weight copying only)')

    # PPO hyperparameters (None = use env-specific default)
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

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='continual_neuroevolution_gymnax')
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--num_eval_episodes', type=int, default=10)

    return parser.parse_args()


def get_hyperparams(args):
    """Get hyperparameters, using env-specific defaults when not specified."""
    env_config = ENV_CONFIGS[args.env]

    return {
        'num_generations': args.num_generations if args.num_generations is not None else env_config['num_generations'],
        'task_interval': args.task_interval if args.task_interval is not None else env_config['task_interval'],
        'num_envs': args.num_envs if args.num_envs is not None else env_config['num_envs'],
        'num_steps': args.num_steps if args.num_steps is not None else env_config['num_steps'],
        'episode_length': args.episode_length if args.episode_length is not None else env_config['episode_length'],
        'learning_rate': args.learning_rate if args.learning_rate is not None else env_config['learning_rate'],
        'gamma': args.gamma if args.gamma is not None else env_config['gamma'],
        'gae_lambda': args.gae_lambda,
        'clip_eps': args.clip_eps,
        'vf_coef': args.vf_coef,
        'ent_coef': args.ent_coef if args.ent_coef is not None else env_config['ent_coef'],
        'num_epochs': args.num_epochs if args.num_epochs is not None else env_config['num_epochs'],
        'num_minibatches': args.num_minibatches if args.num_minibatches is not None else env_config['num_minibatches'],
        'policy_hidden_dims': env_config['policy_hidden_dims'],
        'value_hidden_dims': env_config['value_hidden_dims'],
    }


def main():
    args = parse_args()

    env_name = args.env
    seed = args.seed + args.trial
    trial = args.trial
    pop_size = args.pop_size
    noise_range = args.noise_range

    # Get environment-specific hyperparameters
    hp = get_hyperparams(args)
    num_generations = hp['num_generations']
    task_interval = hp['task_interval']

    print("=" * 60)
    print(f"PBT + PPO on {env_name} (CONTINUAL)")
    print("=" * 60)

    # Create environment
    env, env_params = gymnax.make(env_name)
    env_params = env_params.replace(max_steps_in_episode=hp['episode_length'])
    obs_dim = env.observation_space(env_params).shape[0]
    action_dim = env.action_space(env_params).n

    print(f"  Environment: {env_name}")
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"  Population size: {pop_size}")
    print(f"  PBT interval: {args.pbt_interval} updates")
    print(f"  Exploit fraction: {args.exploit_fraction}")
    print(f"  Perturb factor: {args.perturb_factor}")
    print(f"  PBT mode: {args.pbt_mode}")
    print(f"  Total generations: {num_generations:,}")
    print(f"  Task interval: {task_interval} generations")
    print(f"  Noise range: {noise_range}")
    print(f"  Base hyperparams: num_envs={hp['num_envs']}, num_steps={hp['num_steps']}, "
          f"lr={hp['learning_rate']}, gamma={hp['gamma']}, ent_coef={hp['ent_coef']}")

    # Output directory
    if args.output_dir is None:
        output_dir = os.path.join(
            REPO_ROOT, "projects", "gymnax",
            f"pbt_{args.pbt_mode}_{env_name.replace('-', '_')}_continual",
            f"pop_{pop_size}",
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

    # Create checkpoints directory for per-task checkpoints (used by KL analysis)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Initialize random key
    key = random.key(seed + trial * 1000)

    # Create network templates
    policy_network = PolicyNetwork(hidden_dims=hp['policy_hidden_dims'], action_dim=action_dim)
    value_network = ValueNetwork(hidden_dims=hp['value_hidden_dims'])

    # ================================================================
    # Initialize population with shared optimizer
    # ================================================================
    shared_tx = optax.inject_hyperparams(optax.adam)(learning_rate=hp['learning_rate'])
    population_states = []
    population_hyperparams = []

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

        # Set per-agent learning rate in optimizer state
        p_state.opt_state.hyperparams['learning_rate'] = jnp.array(lr_i)
        v_state.opt_state.hyperparams['learning_rate'] = jnp.array(lr_i)

        population_states.append((p_state, v_state))
        population_hyperparams.append({'learning_rate': lr_i, 'ent_coef': ent_i})

        print(f"  Agent {i}: lr={lr_i:.6f}, ent_coef={ent_i:.6f}")

    # Count parameters
    policy_param_count = sum(p.size for p in jax.tree_util.tree_leaves(population_states[0][0].params))
    value_param_count = sum(p.size for p in jax.tree_util.tree_leaves(population_states[0][1].params))
    print(f"  Policy params: {policy_param_count:,}, Value params: {value_param_count:,}")

    # Stack population into GPU arrays (primary representation during training)
    stacked_pp, stacked_vp = stack_population_params(population_states)
    stacked_p_opt, stacked_v_opt = stack_population_opt_states(population_states)
    # population_states list is only used at PBT steps now

    # Initialize wandb
    config = {
        'env': env_name, 'method': 'pbt', 'mode': 'continual',
        'num_generations': num_generations, 'seed': seed, 'trial': trial,
        'pop_size': pop_size, 'pbt_interval': args.pbt_interval,
        'exploit_fraction': args.exploit_fraction, 'perturb_factor': args.perturb_factor, 'pbt_mode': args.pbt_mode,
        'task_interval': task_interval, 'noise_range': noise_range,
        'num_envs': hp['num_envs'], 'num_steps': hp['num_steps'],
        'base_learning_rate': hp['learning_rate'], 'gamma': hp['gamma'],
        'base_ent_coef': hp['ent_coef'],
        'num_epochs': hp['num_epochs'], 'num_minibatches': hp['num_minibatches'],
        'policy_hidden_dims': hp['policy_hidden_dims'],
        'value_hidden_dims': hp['value_hidden_dims'],
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"pbt_{args.pbt_mode}_{env_name}_continual_pop{pop_size}_trial{trial}", reinit=True)
    wandb.define_metric("generation")
    wandb.define_metric("*", step_metric="generation")

    # Training metrics
    best_reward = -float('inf')
    best_agent_idx = 0
    training_metrics_log = []
    start_time = time.time()

    # Number of updates = number of generations (1 generation = 1 rollout+train cycle)
    timesteps_per_update = hp['num_envs'] * hp['num_steps']
    num_updates = num_generations

    print(f"\nStarting PBT continual training ({num_updates} generations)...")

    # ================================================================
    # Fused rollout + training in single JIT call (minimizes kernel launches)
    # ================================================================
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

    @jax.jit
    def jit_rollout_and_train(key, stacked_pp, stacked_vp, stacked_p_opt, stacked_v_opt, ent_coefs, noise_vector):
        """Fused rollout + training for all agents in a single JIT call."""
        rollout_key, train_key = random.split(key)
        
        # Collect rollouts for all agents (noise_vector shared across population)
        rollout_keys = random.split(rollout_key, pop_size)
        pop_rollouts = jax.vmap(
            lambda k, pp, vp: collect_rollout(
                k, policy_network, pp, value_network, vp,
                env, env_params, _num_envs, _num_steps, noise_vector
            )
        )(rollout_keys, stacked_pp, stacked_vp)
        
        # Train all agents
        train_keys = random.split(train_key, pop_size)
        
        def train_single(key, pp, vp, p_opt, v_opt, rollout, ent_coef):
            # Compute GAE advantages per environment
            rewards_T = rollout['rewards'].T
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
            flat_actions = rollout['actions'].reshape(_batch_size)
            flat_log_probs = rollout['log_probs'].reshape(_batch_size)
            flat_advantages = advantages.reshape(_batch_size)
            flat_returns = returns.reshape(_batch_size)

            def epoch_step(carry, _):
                pp, vp, p_opt, v_opt, key = carry
                key, shuffle_key = random.split(key)
                perm = random.permutation(shuffle_key, _batch_size)

                # Shuffle and reshape into minibatches
                minibatches = {
                    'obs': flat_obs[perm].reshape(_num_mbs, _minibatch_size, -1),
                    'actions': flat_actions[perm].reshape(_num_mbs, _minibatch_size),
                    'log_probs': flat_log_probs[perm].reshape(_num_mbs, _minibatch_size),
                    'advantages': flat_advantages[perm].reshape(_num_mbs, _minibatch_size),
                    'returns': flat_returns[perm].reshape(_num_mbs, _minibatch_size),
                }

                def minibatch_step(carry, mb):
                    pp, vp, p_opt, v_opt = carry

                    def loss_fn(policy_params, value_params):
                        return compute_ppo_loss(
                            policy_params, value_params,
                            policy_network, value_network,
                            mb, _clip_eps, _vf_coef, ent_coef,
                        )

                    (loss, metrics), (p_grads, v_grads) = jax.value_and_grad(
                        loss_fn, argnums=(0, 1), has_aux=True
                    )(pp, vp)

                    p_updates, new_p_opt = shared_tx.update(p_grads, p_opt, pp)
                    pp = optax.apply_updates(pp, p_updates)
                    v_updates, new_v_opt = shared_tx.update(v_grads, v_opt, vp)
                    vp = optax.apply_updates(vp, v_updates)

                    return (pp, vp, new_p_opt, new_v_opt), None

                (pp, vp, p_opt, v_opt), _ = jax.lax.scan(
                    minibatch_step, (pp, vp, p_opt, v_opt), minibatches
                )
                return (pp, vp, p_opt, v_opt, key), None

            (pp, vp, p_opt, v_opt, _), _ = jax.lax.scan(
                epoch_step, (pp, vp, p_opt, v_opt, key), None, length=_num_epochs
            )
            return pp, vp, p_opt, v_opt

        new_pp, new_vp, new_p_opt, new_v_opt = jax.vmap(train_single)(
            train_keys, stacked_pp, stacked_vp, stacked_p_opt, stacked_v_opt,
            pop_rollouts, ent_coefs
        )
        return new_pp, new_vp, new_p_opt, new_v_opt

    # ================================================================
    # Vmapped evaluation (JIT-compatible with noise, replaces Python-loop evaluate)
    # ================================================================
    _batch_size_eval = hp['num_steps'] * hp['num_envs']  # Keep for other uses
    num_eval_eps = args.num_eval_episodes
    max_eval_steps = hp['episode_length']

    @jax.jit
    def jit_evaluate_population(keys, stacked_pp, noise_vector):
        """Evaluate all agents in parallel with noise. Returns (pop_size,) mean rewards."""
        def eval_single(key, policy_params):
            def run_episode(ep_key):
                reset_key, step_key = random.split(ep_key)
                obs, env_state = env.reset(reset_key, env_params)

                def step_fn(carry, _):
                    obs, state, total_reward, done_flag, k = carry
                    noisy_obs = obs + noise_vector
                    logits = policy_network.apply(policy_params, noisy_obs[None])[0]
                    action = jnp.argmax(logits)
                    k, sk = random.split(k)
                    next_obs, next_state, reward, done, _ = env.step(
                        sk, state, action, env_params
                    )
                    total_reward = total_reward + reward * (1.0 - done_flag)
                    done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
                    return (next_obs, next_state, total_reward, done_flag, k), None

                (_, _, total_reward, _, _), _ = jax.lax.scan(
                    step_fn, (obs, env_state, 0.0, 0.0, step_key), None,
                    length=max_eval_steps
                )
                return total_reward

            ep_keys = random.split(key, num_eval_eps)
            rewards = jax.vmap(run_episode)(ep_keys)
            return jnp.mean(rewards)

        return jax.vmap(eval_single)(keys, stacked_pp)

    # Track per-agent fitness for PBT
    population_fitness = np.full(pop_size, -np.inf)

    # Initialize noise vector for Task 0 (zero noise = baseline task)
    noise_vector = jnp.zeros((obs_dim,))
    current_task = 0
    print(f"\n  Task 0 (baseline): no noise")

    # Helper function to save GIFs for current task
    def save_task_gifs(task_idx, noise_vec, task_name, best_policy_params):
        """Save evaluation GIFs for the best agent at end of a task."""
        nonlocal key
        try:
            render_fn = get_render_fn(env_name)
            if render_fn is None:
                return

            num_gifs = 10
            task_gifs_dir = os.path.join(gifs_dir, f"task_{task_idx:02d}_{task_name}")
            os.makedirs(task_gifs_dir, exist_ok=True)

            fig, ax = plt.subplots(figsize=(6, 4))

            for gif_idx in range(num_gifs):
                key, gif_key = random.split(key)
                obs, env_state = env.reset(gif_key, env_params)
                frames = []
                total_reward = 0.0

                for step in range(hp['episode_length']):
                    frame = render_fn(obs, fig, ax, step=step)
                    frames.append(frame)
                    noisy_obs = obs + noise_vec
                    logits = policy_network.apply(best_policy_params, noisy_obs)
                    action = jnp.argmax(logits)
                    gif_key, step_key = random.split(gif_key)
                    obs, env_state, reward, done, _ = env.step(step_key, env_state, action, env_params)
                    total_reward += float(reward)
                    if bool(done):
                        break

                gif_path = os.path.join(task_gifs_dir,
                    f"task{task_idx}_{task_name}_rollout_{gif_idx:02d}_reward{total_reward:.0f}.gif")
                imageio.mimsave(gif_path, frames, fps=30, loop=0)

            plt.close(fig)
            print(f"  Saved {num_gifs} GIFs for task {task_idx} ({task_name})")

        except Exception as e:
            print(f"  Warning: Failed to save GIFs for task {task_idx}: {e}")

    # Cache ent_coefs (updated only after PBT steps)
    ent_coefs = jnp.array([population_hyperparams[i]['ent_coef'] for i in range(pop_size)])

    for update in range(num_updates):
        timestep = (update + 1) * timesteps_per_update

        # ================================================================
        # Check for task switch
        # ================================================================
        if update > 0 and update % task_interval == 0:
            # End-of-task evaluation: evaluate all agents in parallel (use stacked)
            key, eval_rng = random.split(key)
            eval_keys = random.split(eval_rng, pop_size)
            task_eval_rewards_jax = jit_evaluate_population(eval_keys, stacked_pp, noise_vector)
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

            # Save GIFs for the ending task with best agent on this task
            prev_noise_mag = float(jnp.linalg.norm(noise_vector))
            prev_task_name = "baseline" if current_task == 0 else f"noise_{prev_noise_mag:.2f}"
            # Extract best agent's params from stacked arrays
            best_policy_params = jax.tree.map(lambda x: x[task_best_idx], stacked_pp)
            save_task_gifs(current_task, noise_vector, prev_task_name, best_policy_params)

            # Save per-task checkpoint for KL divergence analysis
            task_ckpt_path = os.path.join(checkpoints_dir, f"task_{current_task}.pkl")
            with open(task_ckpt_path, 'wb') as f:
                pickle.dump({
                    'policy_params': jax.device_get(best_policy_params),
                    'task_idx': current_task,
                    'noise_vector': jax.device_get(noise_vector),
                    'generation': update,
                    'best_agent_idx': task_best_idx,
                    'best_reward': float(task_best_reward),
                }, f)
            print(f"    Saved task checkpoint: {task_ckpt_path}")

            current_task += 1
            key, noise_key = random.split(key)
            noise_vector = random.normal(noise_key, (obs_dim,)) * noise_range
            print(f"\n>>> Task {current_task} started at update {update}")
            print(f"  Noise vector: {jax.device_get(noise_vector)}")
            print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")

        # ================================================================
        # Collect rollouts AND train all agents (single fused JIT call)
        # ================================================================
        key, step_rng = random.split(key)
        stacked_pp, stacked_vp, stacked_p_opt, stacked_v_opt = jit_rollout_and_train(
            step_rng, stacked_pp, stacked_vp, stacked_p_opt, stacked_v_opt, ent_coefs, noise_vector
        )

        # ================================================================
        # PBT exploit/explore step
        # ================================================================
        if (update + 1) % args.pbt_interval == 0:
            # Evaluate all agents in parallel with current noise (use already-stacked)
            key, eval_rng = random.split(key)
            eval_keys = random.split(eval_rng, pop_size)
            pop_rewards = jit_evaluate_population(eval_keys, stacked_pp, noise_vector)
            population_fitness = np.array(pop_rewards)

            diversity = compute_diversity_stacked(stacked_pp)
            print(f"\n  PBT step at update {update+1} (Task {current_task}):")
            print(f"    Fitness: {population_fitness}")
            print(f"    Best: agent {np.argmax(population_fitness)} = {np.max(population_fitness):.2f}")
            print(f"    Mean: {np.mean(population_fitness):.2f} | Diversity: {diversity:.4f}")

            # Unstack to list for PBT (only done at PBT steps, not every iteration)
            population_states = unstack_to_population(
                stacked_pp, stacked_vp, stacked_p_opt, stacked_v_opt,
                shared_tx, policy_network, value_network, pop_size
            )

            # Exploit + Explore
            key, pbt_key = random.split(key)
            population_states, population_hyperparams = pbt_exploit_and_explore(
                population_states,
                population_hyperparams,
                population_fitness,
                pbt_key,
                exploit_fraction=args.exploit_fraction,
                perturb_factor=args.perturb_factor,
                mode=args.pbt_mode,
                shared_tx=shared_tx,
            )

            # Restack after PBT
            stacked_pp, stacked_vp = stack_population_params(population_states)
            stacked_p_opt, stacked_v_opt = stack_population_opt_states(population_states)

            # Update cached ent_coefs after PBT hyperparameter changes
            ent_coefs = jnp.array([population_hyperparams[i]['ent_coef'] for i in range(pop_size)])

            diversity_after = compute_diversity_stacked(stacked_pp)

            # Log hyperparams after PBT
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
        # Evaluate best agent periodically
        # ================================================================
        if (update + 1) % args.eval_interval == 0 or update == num_updates - 1:
            # Evaluate all agents in parallel with current noise (use already-stacked)
            key, eval_rng = random.split(key)
            eval_keys = random.split(eval_rng, pop_size)
            eval_rewards_jax = jit_evaluate_population(eval_keys, stacked_pp, noise_vector)
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
                'best_reward': float(best_reward),
                'pop_best_reward': float(pop_best_reward),
                'pop_mean_reward': float(pop_mean_reward),
                'best_agent': best_agent_idx,
                'diversity': diversity,
                'noise_magnitude': float(jnp.linalg.norm(noise_vector)),
                'elapsed_time': elapsed,
            })

            wandb.log({
                'generation': update + 1,
                'timestep': timestep,
                'task': current_task,
                'eval/pop_best_reward': pop_best_reward,
                'eval/pop_mean_reward': pop_mean_reward,
                'eval/best_reward': best_reward,
                'eval/best_agent': best_agent_idx,
                'eval/diversity': diversity,
                'noise_magnitude': float(jnp.linalg.norm(noise_vector)),
            })

            print(f"Update {update+1:5d} | Task {current_task} | Timestep {timestep:10,} | "
                  f"Pop Best: {pop_best_reward:8.2f} (agent {best_agent_idx}) | "
                  f"Pop Mean: {pop_mean_reward:8.2f} | Diversity: {diversity:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s, Best: {best_reward:.2f}")

    # ================================================================
    # Final evaluation of best agent
    # ================================================================
    # Unstack to get final population_states for checkpoint
    population_states = unstack_to_population(
        stacked_pp, stacked_vp, stacked_p_opt, stacked_v_opt,
        shared_tx, policy_network, value_network, pop_size
    )
    print(f"\nFinal evaluation of best agent (agent {best_agent_idx}) on Task {current_task}...")
    best_policy_state, best_value_state = population_states[best_agent_idx]
    final_eval_rewards = []
    for eval_trial in range(10):
        key, eval_key = random.split(key)
        obs, env_state = env.reset(eval_key, env_params)
        trial_reward = 0.0
        for step in range(hp['episode_length']):
            noisy_obs = obs + noise_vector
            logits = policy_network.apply(best_policy_state.params, noisy_obs)
            action = jnp.argmax(logits)
            eval_key, step_key = random.split(eval_key)
            obs, env_state, reward, done, _ = env.step(step_key, env_state, action, env_params)
            trial_reward += float(reward)
            if bool(done):
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

    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"pbt_{env_name.replace('-', '_')}_continual_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'policy_params': best_policy_state.params,
            'value_params': best_value_state.params,
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

    # Save GIFs for the final task
    final_noise_mag = float(jnp.linalg.norm(noise_vector))
    final_task_name = "baseline" if current_task == 0 else f"noise_{final_noise_mag:.2f}"
    save_task_gifs(current_task, noise_vector, final_task_name, best_policy_state.params)

    # Save final task checkpoint for KL divergence analysis
    task_ckpt_path = os.path.join(checkpoints_dir, f"task_{current_task}.pkl")
    with open(task_ckpt_path, 'wb') as f:
        pickle.dump({
            'policy_params': jax.device_get(best_policy_state.params),
            'task_idx': current_task,
            'noise_vector': jax.device_get(noise_vector),
            'generation': num_updates,
            'best_agent_idx': best_agent_idx,
            'best_reward': float(best_reward),
        }, f)
    print(f"Saved final task checkpoint: {task_ckpt_path}")

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
