"""
Population-Based Training (PBT) with PPO on MuJoCo CheetahRun (CONTINUAL).

Uses Brax's PPO infrastructure (same as RL script) for efficient training.
Maintains a population of PPO agents with separate TrainingStates, periodically
exploiting top performers and exploring hyperparameters.

For pop_size=1, this is functionally identical to the RL script.
For pop_size>1, each agent is trained sequentially per epoch using the same
JIT-compiled training functions (compiled once, reused for all agents).

Network architecture (matches GA for fair comparison):
- Policy: 4 hidden layers of 32 neurons each with tanh
- Value: 4 hidden layers of 32 neurons each with tanh

Usage:
    python train_PBT_cheetah_continual.py --env CheetahRun --pop_size 1 --gpus 0
    python train_PBT_cheetah_continual.py --env CheetahRun --pop_size 8 --gpus 0
"""

import argparse
import csv
import functools
import json
import os
import pickle
import sys
import time

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
import flax
from flax import linen
import optax
import wandb
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import losses as ppo_losses
from brax.training import acting, gradients, types
from brax.training.acme import running_statistics, specs
from mujoco_playground import registry, wrapper

from my_brax.ppo_continual_train import TrainingState

_PMAP_AXIS_NAME = 'i'


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


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
# Environment helpers (same as RL script)
# ============================================================================

def create_modified_env(env_name, task_mod='gravity', multiplier=1.0):
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
        raise ValueError(f"Unknown task_mod: {task_mod}")
    env._mjx_model = mjx.put_model(env.mj_model)
    return env


def sample_task_multipliers(num_tasks, default_mult=1.0, low_mult=0.2, high_mult=5.0):
    multiplier_cycle = [default_mult, low_mult, high_mult]
    return [multiplier_cycle[i % 3] for i in range(num_tasks)]


# ============================================================================
# PBT helpers
# ============================================================================

def pbt_exploit_explore(population_states, population_rewards, key, pop_size,
                        exploit_fraction=0.2, mode='full'):
    """PBT exploit/explore step on Brax TrainingStates.

    Exploit: copy params (and normalizer) from top agents to bottom agents.
    Explore: not implemented for Brax optimizer states (LR is fixed).
    """
    if pop_size <= 1:
        return population_states

    num_replace = max(1, int(pop_size * exploit_fraction))
    sorted_indices = np.argsort(population_rewards)
    bottom_indices = sorted_indices[:num_replace]
    top_indices = sorted_indices[-num_replace:]

    for i, bot_idx in enumerate(bottom_indices):
        key, sel_key = random.split(key)
        top_idx = top_indices[int(random.randint(sel_key, (), 0, len(top_indices)))]

        if mode in ('full', 'weights_only'):
            top_state = population_states[top_idx]
            bot_state = population_states[bot_idx]
            population_states[bot_idx] = TrainingState(
                optimizer_state=bot_state.optimizer_state,
                params=top_state.params,
                normalizer_params=top_state.normalizer_params,
                env_steps=bot_state.env_steps,
            )

    return population_states


def compute_param_diversity(population_states, pop_size):
    if pop_size <= 1:
        return 0.0
    flat_params = []
    for i in range(pop_size):
        params = _unpmap(population_states[i].params.policy)
        flat = jnp.concatenate([p.flatten() for p in jax.tree_util.tree_leaves(params)])
        flat_params.append(flat)
    flat_params = jnp.stack(flat_params)
    dists = []
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            dists.append(float(jnp.linalg.norm(flat_params[i] - flat_params[j])))
    return float(np.mean(dists)) if dists else 0.0


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='PBT + PPO on MuJoCo (Continual) - Brax infra')
    parser.add_argument('--env', type=str, default='CheetahRun')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)

    # Continual learning
    parser.add_argument('--num_tasks', type=int, default=30)
    parser.add_argument('--timesteps_per_task', type=int, default=51_200_000)
    parser.add_argument('--task_mod', type=str, default='friction',
                        choices=['gravity', 'friction'])
    parser.add_argument('--gravity_default_mult', type=float, default=1.0)
    parser.add_argument('--gravity_low_mult', type=float, default=0.2)
    parser.add_argument('--gravity_high_mult', type=float, default=5.0)
    parser.add_argument('--friction_default_mult', type=float, default=1.0)
    parser.add_argument('--friction_low_mult', type=float, default=0.2)
    parser.add_argument('--friction_high_mult', type=float, default=5.0)

    # PBT parameters
    parser.add_argument('--pop_size', type=int, default=1)
    parser.add_argument('--pbt_interval', type=int, default=10,
                        help='Run PBT exploit/explore every N epochs')
    parser.add_argument('--exploit_fraction', type=float, default=0.2)
    parser.add_argument('--perturb_factor', type=float, default=0.2)
    parser.add_argument('--pbt_mode', type=str, default='full',
                        choices=['full', 'hp_only', 'weights_only'])

    # PPO hyperparameters (same defaults as RL script for CheetahRun)
    parser.add_argument('--num_envs', type=int, default=256)
    parser.add_argument('--num_eval_envs', type=int, default=128)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--unroll_length', type=int, default=10)
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--num_updates_per_batch', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--entropy_cost', type=float, default=1e-2)
    parser.add_argument('--discounting', type=float, default=0.97)
    parser.add_argument('--reward_scaling', type=float, default=0.1)
    parser.add_argument('--clipping_epsilon', type=float, default=0.3)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--action_repeat', type=int, default=1)
    parser.add_argument('--normalize_observations', type=lambda x: x.lower() == 'true',
                        default=True, metavar='BOOL')
    parser.add_argument('--policy_hidden_sizes', type=str, default='32,32,32,32')
    parser.add_argument('--value_hidden_sizes', type=str, default='32,32,32,32')
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['tanh', 'swish'])

    # Eval and logging
    parser.add_argument('--num_evals_per_task', type=int, default=100)

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='mujoco_evosax')
    parser.add_argument('--run_name', type=str, default=None)

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    env_name = args.env
    seed = args.seed + args.trial
    trial = args.trial
    pop_size = args.pop_size
    task_mod = args.task_mod
    num_tasks = args.num_tasks
    timesteps_per_task = args.timesteps_per_task

    policy_hidden_sizes = tuple(int(x) for x in args.policy_hidden_sizes.split(','))
    value_hidden_sizes = tuple(int(x) for x in args.value_hidden_sizes.split(','))
    activation_fn = jax.nn.tanh if args.activation == 'tanh' else jax.nn.swish

    # Task multipliers
    if task_mod == 'gravity':
        multiplier_list = sample_task_multipliers(
            num_tasks, args.gravity_default_mult, args.gravity_low_mult,
            args.gravity_high_mult)
    else:
        multiplier_list = sample_task_multipliers(
            num_tasks, args.friction_default_mult, args.friction_low_mult,
            args.friction_high_mult)

    # Output directory
    if args.output_dir is None:
        output_dir = os.path.join(
            REPO_ROOT, "projects", "mujoco",
            f"pbt_{args.pbt_mode}_{env_name}_continual_{task_mod}",
            f"trial_{trial}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    gifs_dir = os.path.join(output_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)

    # Set up logging to file
    log_file = os.path.join(output_dir, "train.log")
    sys.stdout = Tee(log_file)

    print("=" * 60)
    print(f"PBT + PPO on {env_name} (CONTINUAL, Brax infra)")
    print("=" * 60)

    # ================================================================
    # Device setup (same as ppo_continual_train.py)
    # ================================================================
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    device_count = local_devices_to_use * process_count

    num_envs = args.num_envs
    batch_size = args.batch_size
    num_minibatches = args.num_minibatches
    num_updates_per_batch = args.num_updates_per_batch
    unroll_length = args.unroll_length
    episode_length = args.episode_length
    action_repeat = args.action_repeat

    assert num_envs % device_count == 0, \
        f"num_envs ({num_envs}) must be divisible by device_count ({device_count})"
    assert batch_size * num_minibatches % num_envs == 0, \
        f"batch_size*num_minibatches ({batch_size*num_minibatches}) must be divisible by num_envs ({num_envs})"

    # ================================================================
    # Compute training schedule (same as ppo_continual_train.py)
    # ================================================================
    env_step_per_training_step = (
        batch_size * unroll_length * num_minibatches * action_repeat)
    num_evals_per_task = args.num_evals_per_task
    num_training_steps_per_epoch = int(np.ceil(
        timesteps_per_task / (num_evals_per_task * env_step_per_training_step)))
    num_training_epochs = num_evals_per_task

    print(f"\nConfig:")
    print(f"  Population size: {pop_size}")
    print(f"  PBT interval: {args.pbt_interval} epochs")
    print(f"  PBT mode: {args.pbt_mode}")
    print(f"  Tasks: {num_tasks}, Timesteps/task: {timesteps_per_task:,}")
    print(f"  Num envs: {num_envs}, Batch size: {batch_size}")
    print(f"  Unroll length: {unroll_length}, Num minibatches: {num_minibatches}")
    print(f"  Env steps per training step: {env_step_per_training_step:,}")
    print(f"  Training steps per epoch: {num_training_steps_per_epoch}")
    print(f"  Epochs per task: {num_training_epochs}")
    print(f"  Policy hidden: {policy_hidden_sizes}, Value hidden: {value_hidden_sizes}")
    print(f"  LR: {args.learning_rate}, Entropy: {args.entropy_cost}")
    print(f"  Multipliers: {[f'{m:.2f}' for m in multiplier_list]}")

    # ================================================================
    # Create first environment and Brax infrastructure
    # ================================================================
    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_policy, key_value = jax.random.split(global_key)

    first_env = create_modified_env(env_name, task_mod, multiplier_list[0])
    wrapped_env = wrapper.wrap_for_brax_training(
        first_env, episode_length=episode_length, action_repeat=action_repeat)

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(
        key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    reset_fn = jax.pmap(wrapped_env.reset, axis_name=_PMAP_AXIS_NAME)
    env_state = reset_fn(key_envs)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

    # ================================================================
    # Create PPO networks (identical to RL script)
    # ================================================================
    normalize = running_statistics.normalize if args.normalize_observations \
        else lambda x, y: x

    ppo_network = ppo_networks.make_ppo_networks(
        obs_shape, wrapped_env.action_size,
        preprocess_observations_fn=normalize,
        policy_hidden_layer_sizes=policy_hidden_sizes,
        value_hidden_layer_sizes=value_hidden_sizes,
        activation=activation_fn,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    # ================================================================
    # Optimizer (identical to ppo_continual_train.py)
    # ================================================================
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(learning_rate=args.learning_rate),
    )

    # ================================================================
    # Loss function (identical to ppo_continual_train.py)
    # ================================================================
    loss_fn = functools.partial(
        ppo_losses.compute_ppo_loss,
        ppo_network=ppo_network,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        reward_scaling=args.reward_scaling,
        gae_lambda=args.gae_lambda,
        clipping_epsilon=args.clipping_epsilon,
        normalize_advantage=True,
    )
    loss_and_pgrad_fn = gradients.loss_and_pgrad(
        loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)

    # ================================================================
    # Training step functions (identical to ppo_continual_train.py)
    # ================================================================
    def minibatch_step(carry, data, normalizer_params):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), grads = loss_and_pgrad_fn(
            params, normalizer_params, data, key_loss)
        params_update, optimizer_state = optimizer.update(
            grads, optimizer_state, params)
        params = optax.apply_updates(params, params_update)
        return (optimizer_state, params, key), metrics

    def sgd_step(carry, unused_t, data, normalizer_params):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)
        def convert_data(x):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x
        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key), metrics

    def training_step(carry, unused_t, env, reset_fn_local):
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)
        policy = make_policy((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        ))
        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env, current_state, policy, current_key, unroll_length,
                extra_fields=('truncation', 'episode_metrics', 'episode_done'),
            )
            return (next_state, next_key), data
        (state, _), data = jax.lax.scan(
            f, (state, key_generate_unroll), (),
            length=batch_size * num_minibatches // num_envs,
        )
        data = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
        normalizer_params = running_statistics.update(
            training_state.normalizer_params, data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step, data=data, normalizer_params=normalizer_params),
            (training_state.optimizer_state, training_state.params, key_sgd),
            (), length=num_updates_per_batch,
        )
        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
        )
        return (new_training_state, state, new_key), metrics

    # ================================================================
    # Initialize population of TrainingStates
    # ================================================================
    obs_shape_spec = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')),
        env_state.obs)

    population_states = []    # List of pmap-replicated TrainingStates
    population_env_states = []  # List of pmap-replicated env states

    for i in range(pop_size):
        key_p_i, key_v_i, local_key = jax.random.split(local_key, 3)
        # Use different init keys per agent for diversity
        if i == 0:
            # First agent uses the global keys (matches single-agent RL)
            init_params = ppo_losses.PPONetworkParams(
                policy=ppo_network.policy_network.init(key_policy),
                value=ppo_network.value_network.init(key_value),
            )
        else:
            init_params = ppo_losses.PPONetworkParams(
                policy=ppo_network.policy_network.init(key_p_i),
                value=ppo_network.value_network.init(key_v_i),
            )
        ts = TrainingState(
            optimizer_state=optimizer.init(init_params),
            params=init_params,
            normalizer_params=running_statistics.init_state(obs_shape_spec),
            env_steps=jnp.array(0),
        )
        ts = jax.device_put_replicated(
            ts, jax.local_devices()[:local_devices_to_use])
        population_states.append(ts)

        # Each agent gets its own env state
        agent_key_envs = jax.random.split(
            jax.random.fold_in(key_env, i),
            num_envs // process_count)
        agent_key_envs = jnp.reshape(
            agent_key_envs,
            (local_devices_to_use, -1) + agent_key_envs.shape[1:])
        population_env_states.append(reset_fn(agent_key_envs))

    # Count params
    sample_params = _unpmap(population_states[0].params.policy)
    policy_param_count = sum(
        p.size for p in jax.tree_util.tree_leaves(sample_params))
    sample_vparams = _unpmap(population_states[0].params.value)
    value_param_count = sum(
        p.size for p in jax.tree_util.tree_leaves(sample_vparams))
    print(f"  Policy params: {policy_param_count:,}, "
          f"Value params: {value_param_count:,}")

    # ================================================================
    # Initialize wandb
    # ================================================================
    config = {
        'env': env_name, 'method': 'pbt', 'mode': 'continual',
        'pop_size': pop_size, 'pbt_interval': args.pbt_interval,
        'pbt_mode': args.pbt_mode, 'exploit_fraction': args.exploit_fraction,
        'task_mod': task_mod, 'num_tasks': num_tasks,
        'timesteps_per_task': timesteps_per_task,
        'num_envs': num_envs, 'batch_size': batch_size,
        'unroll_length': unroll_length, 'num_minibatches': num_minibatches,
        'num_updates_per_batch': num_updates_per_batch,
        'learning_rate': args.learning_rate, 'entropy_cost': args.entropy_cost,
        'discounting': args.discounting, 'reward_scaling': args.reward_scaling,
        'policy_hidden_sizes': policy_hidden_sizes,
        'value_hidden_sizes': value_hidden_sizes, 'activation': args.activation,
        'num_evals_per_task': num_evals_per_task,
        'seed': seed, 'trial': trial,
        'task_multipliers': multiplier_list,
    }
    run_name = args.run_name or \
        f"pbt_{args.pbt_mode}_{env_name}_continual_{task_mod}_trial{trial}"
    wandb.init(project=args.wandb_project, config=config,
               name=run_name, reinit='finish_previous')
    wandb.define_metric("generation")
    wandb.define_metric("*", step_metric="generation")

    # ================================================================
    # Training loop
    # ================================================================
    start_time = time.time()
    best_reward_overall = -float('inf')
    best_reward_per_task = {}
    best_agent_idx = 0
    training_metrics_list = []

    print(f"\nStarting PBT continual training...")
    print(f"  {num_tasks} tasks x {num_training_epochs} epochs/task "
          f"= {num_tasks * num_training_epochs} total epochs")

    for task_idx, multiplier in enumerate(multiplier_list):
        print(f"\n{'='*60}")
        print(f"Task {task_idx}/{num_tasks}: {task_mod}={multiplier:.2f}")
        print(f"{'='*60}")

        # Create envs for this task
        train_env = create_modified_env(env_name, task_mod, multiplier)
        eval_env = create_modified_env(env_name, task_mod, multiplier)
        wrapped_train_env = wrapper.wrap_for_brax_training(
            train_env, episode_length=episode_length,
            action_repeat=action_repeat)
        wrapped_eval_env = wrapper.wrap_for_brax_training(
            eval_env, episode_length=episode_length,
            action_repeat=action_repeat)

        # Reset env states for all agents
        reset_fn_task = jax.pmap(
            wrapped_train_env.reset, axis_name=_PMAP_AXIS_NAME)
        for i in range(pop_size):
            agent_key_envs = jax.random.split(
                jax.random.fold_in(
                    jax.random.fold_in(key_env, task_idx), i),
                num_envs // process_count)
            agent_key_envs = jnp.reshape(
                agent_key_envs,
                (local_devices_to_use, -1) + agent_key_envs.shape[1:])
            population_env_states[i] = reset_fn_task(agent_key_envs)

        # Create training epoch for this environment
        # (JIT compiled once per task, reused for all agents and epochs)
        def training_epoch(training_state, state, key):
            training_step_fn = functools.partial(
                training_step, env=wrapped_train_env,
                reset_fn_local=reset_fn_task)
            (training_state, state, _), loss_metrics = jax.lax.scan(
                training_step_fn, (training_state, state, key), (),
                length=num_training_steps_per_epoch,
            )
            loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
            return training_state, state, loss_metrics

        training_epoch_pmap = jax.pmap(
            training_epoch, axis_name=_PMAP_AXIS_NAME,
            donate_argnums=(0, 1))

        # Evaluator for this task
        evaluator = acting.Evaluator(
            wrapped_eval_env,
            functools.partial(make_policy, deterministic=True),
            num_eval_envs=args.num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=jax.random.fold_in(eval_key, task_idx),
        )

        # Training epochs for this task
        for epoch in range(num_training_epochs):
            t = time.time()

            # Train each agent for one epoch
            for agent_idx in range(pop_size):
                epoch_key, local_key = jax.random.split(local_key)
                epoch_keys = jax.random.split(
                    epoch_key, local_devices_to_use)
                (population_states[agent_idx],
                 population_env_states[agent_idx], _) = \
                    training_epoch_pmap(
                        population_states[agent_idx],
                        population_env_states[agent_idx],
                        epoch_keys)

            epoch_time = time.time() - t
            generation = task_idx * num_evals_per_task + epoch + 1

            # PBT exploit/explore at intervals
            is_pbt_step = (pop_size > 1
                           and (epoch + 1) % args.pbt_interval == 0)

            if is_pbt_step:
                agent_rewards = []
                for agent_idx in range(pop_size):
                    params_i = _unpmap((
                        population_states[agent_idx].normalizer_params,
                        population_states[agent_idx].params.policy,
                        population_states[agent_idx].params.value,
                    ))
                    metrics_i = evaluator.run_evaluation(params_i, {})
                    agent_rewards.append(float(
                        metrics_i.get('eval/episode_reward', 0)))

                agent_rewards_np = np.array(agent_rewards)
                best_agent_idx = int(np.argmax(agent_rewards_np))
                pop_best_reward = agent_rewards_np[best_agent_idx]
                pop_mean_reward = float(np.mean(agent_rewards_np))
                diversity = compute_param_diversity(
                    population_states, pop_size)

                print(f"  PBT step at epoch {epoch+1}: "
                      f"Best={pop_best_reward:.2f} "
                      f"(agent {best_agent_idx}), "
                      f"Mean={pop_mean_reward:.2f}, "
                      f"Diversity={diversity:.4f}", flush=True)

                # Exploit/explore
                local_key, pbt_key = jax.random.split(local_key)
                population_states = pbt_exploit_explore(
                    population_states, agent_rewards_np, pbt_key,
                    pop_size,
                    exploit_fraction=args.exploit_fraction,
                    mode=args.pbt_mode,
                )

                wandb.log({
                    'pbt/fitness_max': pop_best_reward,
                    'pbt/fitness_mean': pop_mean_reward,
                    'pbt/diversity': diversity,
                    'pbt/best_agent': best_agent_idx,
                    'generation': generation,
                    'task': task_idx,
                }, step=generation)

            # Evaluate best (or only) agent for logging
            best_params = _unpmap((
                population_states[best_agent_idx].normalizer_params,
                population_states[best_agent_idx].params.policy,
                population_states[best_agent_idx].params.value,
            ))
            metrics = evaluator.run_evaluation(
                best_params, {})
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready()
                if hasattr(x, 'block_until_ready') else x,
                metrics)

            reward = float(metrics.get('eval/episode_reward', 0))
            if task_idx not in best_reward_per_task:
                best_reward_per_task[task_idx] = -float('inf')
            if reward > best_reward_per_task[task_idx]:
                best_reward_per_task[task_idx] = reward
            if reward > best_reward_overall:
                best_reward_overall = reward

            elapsed = time.time() - start_time
            global_step = int(_unpmap(
                population_states[best_agent_idx].env_steps))

            log_data = {
                'generation': generation,
                'global_step': global_step,
                'task': task_idx,
                'multiplier': multiplier,
                'eval/episode_reward': reward,
                'fitness/best': reward,
                'fitness/best_task': best_reward_per_task[task_idx],
                'fitness/best_overall': best_reward_overall,
                'training/epoch_time': epoch_time,
            }
            log_data.update(metrics)
            wandb.log(log_data, step=generation)

            print(f"Task {task_idx+1} Gen {generation:>5d} | "
                  f"Reward: {reward:8.2f} | "
                  f"Best Task: {best_reward_per_task[task_idx]:8.2f} | "
                  f"Best Overall: {best_reward_overall:8.2f} | "
                  f"{task_mod}: {multiplier:6.2f} | "
                  f"Time: {elapsed:6.1f}s", flush=True)

            training_metrics_list.append({
                'generation': generation,
                'step': global_step,
                'task': task_idx,
                'multiplier': multiplier,
                'reward': reward,
                'best_task_reward': best_reward_per_task[task_idx],
                'best_overall_reward': best_reward_overall,
                'elapsed_time': elapsed,
            })

        # ============================================================
        # End of task: checkpoint + GIFs
        # ============================================================
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"task_{task_idx:02d}_{task_mod}_{multiplier:.2f}.pkl".replace(
                ".", "p", 1))
        best_params_save = _unpmap((
            population_states[best_agent_idx].normalizer_params,
            population_states[best_agent_idx].params.policy,
            population_states[best_agent_idx].params.value,
        ))
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'normalizer_params': best_params_save[0],
                'policy_params': best_params_save[1],
                'value_params': best_params_save[2],
                'task_idx': task_idx,
                'multiplier': multiplier,
                'best_reward': best_reward_per_task.get(task_idx, 0),
            }, f)
        print(f"  Checkpoint saved: {checkpoint_path}")

        # Save GIFs for this task
        try:
            jit_inference_fn = jax.jit(
                functools.partial(make_policy, deterministic=True)(
                    best_params_save))
            task_label = f"{task_mod}_{multiplier:.2f}".replace(".", "p")
            task_gifs_dir = os.path.join(
                gifs_dir, f"task_{task_idx:02d}_{task_label}")
            os.makedirs(task_gifs_dir, exist_ok=True)

            jit_reset = jax.jit(train_env.reset)
            jit_step = jax.jit(train_env.step)

            for gif_idx in range(5):
                gif_key = jax.random.key(task_idx * 1000 + gif_idx)
                state = jit_reset(gif_key)
                rollout = [state]
                total_reward = 0.0
                for _ in range(episode_length):
                    gif_key, act_key = jax.random.split(gif_key)
                    action, _ = jit_inference_fn(state.obs, act_key)
                    state = jit_step(state, action)
                    rollout.append(state)
                    total_reward += float(state.reward)
                    if state.done:
                        break
                images = train_env.render(
                    rollout[::2], height=240, width=320, camera="side")
                gif_path = os.path.join(
                    task_gifs_dir,
                    f"trajectory_{gif_idx:02d}_"
                    f"reward{total_reward:.0f}.gif")
                imageio.mimsave(gif_path, images, fps=30, loop=0)
            print(f"  Saved 5 GIFs for task {task_idx}")
        except Exception as e:
            print(f"  Warning: GIF saving failed: {e}")

    # ================================================================
    # Training complete
    # ================================================================
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s, "
          f"Best: {best_reward_overall:.2f}")

    # Save training metrics CSV
    if training_metrics_list:
        csv_path = os.path.join(output_dir, "training_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=training_metrics_list[0].keys())
            writer.writeheader()
            writer.writerows(training_metrics_list)
        print(f"  Training metrics saved to: {csv_path}")

    # Save final checkpoint
    ckpt_path = os.path.join(
        output_dir,
        f"pbt_{env_name}_continual_{task_mod}_best.pkl")
    final_params = _unpmap((
        population_states[best_agent_idx].normalizer_params,
        population_states[best_agent_idx].params.policy,
        population_states[best_agent_idx].params.value,
    ))
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'normalizer_params': final_params[0],
            'policy_params': final_params[1],
            'value_params': final_params[2],
            'best_reward': best_reward_overall,
            'best_agent_idx': best_agent_idx,
            'config': config,
        }, f)
    print(f"  Final checkpoint saved: {ckpt_path}")

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
