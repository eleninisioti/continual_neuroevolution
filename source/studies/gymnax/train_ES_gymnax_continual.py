"""
Train a distribution-based NE method on Gymnax environments (CONTINUAL).

Task changes every 200 generations by adding observation noise.
Noise vector = normal(obs_size) * noise_range

Two methods, one runner, chosen with `--algo` -- see train_ES_gymnax.py for why
they share a trainer:

    openes   centered ranks + Adam (default).
    nes      `source/algorithms/ne/es.py` -- standardized fitness + SGD.

The continual setting is where the two are expected to come apart. Centered
ranks discard how much better a perturbation was, so the generation after a task
switch takes the same size step as any other generation; Adam carries a
second-moment estimate across the switch that the new task's landscape did not
produce. NES has neither property.

Supports CartPole-v1, Acrobot-v1, MountainCar-v0.

Usage:
    python train_ES_gymnax_continual.py --env CartPole-v1 --gpus 0
    python train_ES_gymnax_continual.py --env Acrobot-v1 --gpus 0 --algo nes
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# One dirname deeper since 2026-09-08: the trainers moved from
# `source/<suite>/` into `source/studies/<suite>/`. The runner
# invokes them as SCRIPTS, so `source` is importable only via
# this insert -- a short walk here is a ModuleNotFoundError at
# launch, not a subtle one.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Shared across every trainer -- see source/.
from source.algorithms.networks import MLPPolicy, create_policy_network, get_flat_params, unflatten_params
from source.algorithms.networks import ACTIVATIONS
from source.algorithms.rl.ppo import GYMNAX_POLICY_ACTIVATION
from source.metrics import plasticity
from source.metrics.weight_stats import population_weight_stats, weight_stats
from source.algorithms.rl import redo as redo_mod
from source.utils.runtime import Tee, _get_gpu_arg

_gpu_arg = _get_gpu_arg()
if _gpu_arg:
    os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_arg
    print(f"Setting CUDA_VISIBLE_DEVICES={_gpu_arg}")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from jax import random, flatten_util
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import flax.linen as nn
import optax
import gymnax
import time
import pickle
import wandb
import numpy as np
import imageio
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from source.metrics.evaluation_metrics import THRESHOLD_SETS
from source.envs.gymnax_classic import task_noise_vectors as make_task_noise_vectors
from source.utils.run_artifacts import save_eval_artifacts, save_training_metrics
from source.utils.task_sequence import (
    GYMNAX_PHYSICS_TASKS, action_flip_sequence, cycle_task_sequence,
    physics_mult_sequence)
from source.envs.gymnax_classic import (
    DEEPSEA_ENV_NAMES, DEEPSEA_SIZES, FlippedParams, apply_physics,
    make_gymnax_env, wrap_actions)

# The same set, under the same name, as train_GA_gymnax_continual.py and
# train_DNS_gymnax_continual.py, so an ES row of a continual table is scored
# against what a GA row is scored against.
SOLVED_THRESHOLDS = THRESHOLD_SETS['rebuttal']


# Behavioural-diversity tracking: an observer over the sampled population,
# never part of the ES update.
from source.metrics.aurora import episode_relative_indices, subsample_indices
from source.metrics.behaviour_descriptors import (
    BehaviourConfig,
    average_over_evals,
    rollout_behaviour,
)
from source.metrics.zero_shot import attach as attach_zero_shot
# NES's sigma/lr and the two-line constructor are shared with the stationary
# trainer, so "which settings did the NES arm use" has one answer for both.
from source.studies.gymnax.es_algorithms import build_strategy, resolve_algo_config
from source.metrics.behaviour_tracking import (
    PopulationDiversityTracker,
    add_diversity_args,
    continual_snapshot_gens,
    summarise,
)


# ============================================================================
# Logging Helper
# ============================================================================


# ============================================================================
# Policy Network (Discrete Actions)
# ============================================================================





# ============================================================================
# Environment Configs
# ============================================================================

# sigma / learning_rate are the per-env sweep winners from the noncontinual
# trainer (train_ES_gymnax.py), carried over here so the two settings agree.
# The old sigma=0.5 / lr=0.01 was the worst of the 20 pairs swept and solved
# 0/5 seeds on all three tasks: sigma=0.5 lands most of the population in the
# same failure mode, so the centered ranks are mostly ties, and lr=0.01 cannot
# act on what signal survives. That matters more here than in the
# noncontinual setting, since every sub-task switch asks ES to re-adapt.
ENV_CONFIGS = {
    "CartPole-v1": {
        "num_generations": 2000,  # 10 tasks x 200 gens
        "pop_size": 512,
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 1,
        "sigma": 0.2,
        "learning_rate": 0.2,
        "task_interval": 200,
        "num_tasks": 10,
    },
    "Acrobot-v1": {
        "num_generations": 2000,
        "pop_size": 512,
        # (16, 16), not (32, 32): the policy is pinned across every
        # method compared on this env (source/algorithms/networks.py
        # POLICY_ARCH). It read (32, 32) here while every run on disk
        # was launched with an explicit --hidden_dims 16 16.
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 1,
        "sigma": 0.2,
        "learning_rate": 0.2,
        "task_interval": 200,
        "num_tasks": 10,
    },
    # The sparse-reward case, where the old setting was starkest: nearly every
    # perturbation timed out at -500, so the ranks really were pure noise.
    "MountainCar-v0": {
        "num_generations": 2000,
        "pop_size": 512,
        # (16, 16), not (32, 32): the policy is pinned across every
        # method compared on this env (source/algorithms/networks.py
        # POLICY_ARCH). It read (32, 32) here while every run on disk
        # was launched with an explicit --hidden_dims 16 16.
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 1,
        "sigma": 0.1,
        "learning_rate": 0.1,
        "task_interval": 200,
        "num_tasks": 10,
    },
}

# DeepSea<N>: one row a step, so the episode is N steps; everything else as
# the classic tasks. See DeepSeaEnv in source/envs/gymnax_classic.py.
for _n in DEEPSEA_SIZES:
    ENV_CONFIGS[f"DeepSea{_n}-bsuite"] = {
        "num_generations": 2000,
        "pop_size": 512,
        "hidden_dims": (16, 16),
        "episode_length": _n,
        "num_evals": 1,
        "sigma": 0.2,
        "learning_rate": 0.2,
        "task_interval": 200,
        "num_tasks": 10,
    }


# ============================================================================
# GIF Rendering
# ============================================================================

def render_cartpole_frame(obs, fig=None, ax=None, step=None):
    """Render a single CartPole frame from observation."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.clear()
    
    x, x_dot, theta, theta_dot = obs[0], obs[1], obs[2], obs[3]
    
    # Cart dimensions
    cart_width = 0.4
    cart_height = 0.2
    pole_length = 0.6
    
    # Draw track
    ax.axhline(y=0, color='gray', linewidth=2)
    
    # Draw cart
    cart_x = float(x) - cart_width / 2
    cart = Rectangle((cart_x, 0), cart_width, cart_height, color='blue')
    ax.add_patch(cart)
    
    # Draw pole
    pole_x_end = float(x) + pole_length * np.sin(float(theta))
    pole_y_end = cart_height + pole_length * np.cos(float(theta))
    ax.plot([float(x), pole_x_end], [cart_height, pole_y_end], 'r-', linewidth=4)
    
    # Draw pole tip
    ax.plot(pole_x_end, pole_y_end, 'ro', markersize=8)
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f'CartPole - Step {step}' if step is not None else 'CartPole')
    
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    
    return image, fig, ax


def render_acrobot_frame(obs, fig=None, ax=None, step=None):
    """Render a single Acrobot frame from observation."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.clear()
    
    cos1, sin1, cos2, sin2, _, _ = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]
    
    # Link lengths
    l1, l2 = 1.0, 1.0
    
    # Joint positions
    p1 = [float(l1 * sin1), -float(l1 * cos1)]
    p2 = [p1[0] + float(l2 * sin2) * float(cos1) + float(l2 * cos2) * float(sin1),
          p1[1] - float(l2 * sin2) * float(sin1) + float(l2 * cos2) * float(cos1)]
    
    # Simplified joint 2 calculation
    theta1 = np.arctan2(float(sin1), float(cos1))
    theta2 = np.arctan2(float(sin2), float(cos2))
    p2 = [p1[0] + l2 * np.sin(theta1 + theta2),
          p1[1] - l2 * np.cos(theta1 + theta2)]
    
    # Draw links
    ax.plot([0, p1[0]], [0, p1[1]], 'b-', linewidth=4)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=4)
    
    # Draw joints
    ax.plot(0, 0, 'ko', markersize=10)
    ax.plot(p1[0], p1[1], 'ko', markersize=8)
    ax.plot(p2[0], p2[1], 'go', markersize=8)
    
    # Draw goal line
    ax.axhline(y=l1, color='green', linestyle='--', alpha=0.5)
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title(f'Acrobot - Step {step}' if step is not None else 'Acrobot')
    
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    
    return image, fig, ax


def render_mountaincar_frame(obs, fig=None, ax=None, step=None):
    """Render a single MountainCar frame from observation."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.clear()
    
    position, velocity = float(obs[0]), float(obs[1])
    
    # Draw mountain
    xs = np.linspace(-1.2, 0.6, 100)
    ys = np.sin(3 * xs) * 0.45 + 0.55
    ax.fill_between(xs, 0, ys, color='green', alpha=0.3)
    ax.plot(xs, ys, 'g-', linewidth=2)
    
    # Draw goal
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
    ax.plot(0.5, np.sin(3 * 0.5) * 0.45 + 0.55, 'r*', markersize=15)
    
    # Draw car
    car_y = np.sin(3 * position) * 0.45 + 0.55
    ax.plot(position, car_y, 'bo', markersize=12)
    
    ax.set_xlim(-1.3, 0.7)
    ax.set_ylim(0, 1.2)
    ax.set_title(f'MountainCar - Step {step}' if step is not None else 'MountainCar')
    
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    
    return image, fig, ax


def get_render_fn(env_name):
    if 'CartPole' in env_name:
        return render_cartpole_frame
    elif 'Acrobot' in env_name:
        return render_acrobot_frame
    elif 'MountainCar' in env_name:
        return render_mountaincar_frame
    return render_cartpole_frame


# ============================================================================
# Scoring Function with Observation Noise
# ============================================================================

def make_scoring_fn(env, env_params, policy, param_template, episode_length,
                    num_evals=10, behaviour_cfg=None, traj_steps=10):
    """Create scoring function that accepts noise_vector for continual learning.
    
    Evaluates each individual with num_evals trials:
    - Returns fitness from FIRST trial only (for selection)
    - Returns mean fitness across all trials (for logging/tracking)
    """
    
    traj_indices = subsample_indices(episode_length, traj_steps)
    num_traj_steps = int(traj_indices.shape[0])
    track = behaviour_cfg is not None
    def evaluate_single(flat_params, eval_key, noise_vector, ep):
        params = unflatten_params(flat_params, param_template)
        obs, state = env.reset(eval_key, ep)
        
        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
            # Add noise to observation for continual learning
            noisy_obs = obs + noise_vector
            logits = policy.apply(params, noisy_obs)
            action = jnp.argmax(logits)  # Deterministic for evaluation
            
            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(step_key, state, action, ep)
            
            total_reward = total_reward + reward * (1.0 - done_flag)
            valid = 1.0 - done_flag
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))

            per_step = (noisy_obs, action, valid) if track else None
            return (next_obs, next_state, total_reward, done_flag, key), per_step

        key = eval_key
        (_, _, total_reward, _, _), per_step = jax.lax.scan(
            step_fn, (obs, state, 0.0, 0.0, key), None, length=episode_length
        )
        if not track:
            return total_reward

        all_obs, all_actions, valid = per_step
        # Descriptors are built from the noisy observations the policy acted
        # on, which is what defines the sub-task.
        behaviour = rollout_behaviour(all_obs, all_actions, valid, behaviour_cfg)
        return (total_reward,
                all_obs[episode_relative_indices(valid, num_traj_steps)],
                behaviour)
    
    vmapped_eval = jax.vmap(evaluate_single, in_axes=(0, 0, None, None))
    
    @jax.jit
    def scoring_fn(flat_genotypes, key, noise_vector, task_env_params=None):
        """Returns (fitness_for_selection, mean_fitness_for_logging).

        `task_env_params` is the body of the CURRENT sub-task under
        `--task_type param`, where a sub-task rescales the physics rather than
        offsetting the observation. Left None -- every noise run, and every run
        made before 2026-09-08 -- the jitted function closes over the stock
        body it was built with, unchanged.
        """
        ep = env_params if task_env_params is None else task_env_params
        pop_size = flat_genotypes.shape[0]
        
        # Always evaluate with num_evals trials per individual
        all_keys = random.split(key, pop_size * num_evals)
        flat_params_repeated = jnp.repeat(flat_genotypes, num_evals, axis=0)
        if track:
            all_fitnesses, all_traj, all_behaviour = vmapped_eval(
                flat_params_repeated, all_keys, noise_vector, ep)
        else:
            all_fitnesses = vmapped_eval(flat_params_repeated, all_keys, noise_vector, ep)
        all_fitnesses = all_fitnesses.reshape(pop_size, num_evals)

        # Fitness for selection: use FIRST trial only
        fitnesses = all_fitnesses[:, 0]

        # Mean fitness for logging/tracking
        mean_fitnesses = jnp.mean(all_fitnesses, axis=1)

        if not track:
            return fitnesses, mean_fitnesses, None

        observations = all_traj.reshape(
            pop_size, num_evals, num_traj_steps, -1)[:, 0]
        return fitnesses, mean_fitnesses, {
            "observations": observations,
            "behaviour": average_over_evals(all_behaviour, pop_size, num_evals),
        }
    
    return scoring_fn


def rollout_for_gif(env, env_params, policy, flat_params, param_template, episode_length, key, noise_vector):
    """Rollout for GIF generation with noise."""
    params = unflatten_params(flat_params, param_template)
    obs, state = env.reset(key, env_params)
    
    obs_list = [np.array(obs)]
    total_reward = 0.0
    
    for _ in range(episode_length):
        noisy_obs = obs + noise_vector
        logits = policy.apply(params, noisy_obs)
        action = int(jnp.argmax(logits))
        
        key, step_key = random.split(key)
        obs, state, reward, done, _ = env.step(step_key, state, action, env_params)
        total_reward += float(reward)
        obs_list.append(np.array(obs))
        
        if bool(done):
            break
    
    return obs_list, total_reward


# ============================================================================
# Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='OpenES / NES on Gymnax (Continual)')
    parser.add_argument('--algo', type=str, default='openes',
                        choices=['openes', 'nes'],
                        help='openes: centered ranks + Adam. '
                             'nes: standardized fitness + SGD (search gradient).')
    parser.add_argument('--std_lr', type=float, default=0.0,
                        help='NES only: separable-NES learning rate for the '
                             'per-coordinate search width (0 = fixed sigma).')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0',
                                 *DEEPSEA_ENV_NAMES])
    parser.add_argument('--num_generations', type=int, default=None)
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_evals', type=int, default=None)
    parser.add_argument('--report_episodes', type=int, default=10,
                        help='Episodes behind the REPORTED numbers '
                             '(`centroid_fitness`, `elite_eval_fitness`), '
                             'which are deliberately NOT the numbers the '
                             'search selects on. Selection takes the FIRST of '
                             'the --num_evals rollouts, and the fitness '
                             'columns are a MAX over the population, so they '
                             'are an optimistically biased estimator -- biased '
                             'by a different amount per method (measured on '
                             'the sigma=1.0 tree: +5 for the GA, +53 for DNS), '
                             'which therefore does not cancel in a comparison. '
                             'These two columns are scored on fresh keys at '
                             'the same 10 episodes the RL trainer already uses '
                             'for `mean_reward` (--num_eval_episodes), so both '
                             'families report one estimator. Costs '
                             '2 x report_episodes rollouts a generation '
                             'against pop x num_evals, and never feeds '
                             'selection.')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=None,
                        help='Policy hidden layer sizes. Must match across methods for '
                             'their numbers to be comparable (default: env config).')
    parser.add_argument('--task_interval', type=int, default=200)
    parser.add_argument('--task_period', type=int, default=0,
                        help='Revisit sub-tasks: after this many distinct ones '
                             'the sequence cycles, so a 20-sub-task run at '
                             'period 10 sees each sub-task twice. 0 (default) '
                             'means every sub-task is new, which is every run '
                             'made before this flag existed.')
    parser.add_argument('--task_type', type=str, default='noise',
                        choices=['noise', 'param', 'actions'],
                        help='What a sub-task IS: an observation offset (noise) or a '
                             'rescaled physics group (param). Added 2026-09-08 -- this '
                             'trainer had no param branch, so an ES/NES arm queued into '
                             'a param sweep silently ran the NOISE experiment and every '
                             'other method in that tree faced different sub-tasks.')
    parser.add_argument('--param_name', type=str, default=None,
                        help='Which physics group a param sub-task rescales -- a key of '
                             'PHYSICS_PARAMS[env] in source/envs/gymnax_classic.py. '
                             'Default: the env entry in GYMNAX_PHYSICS_TASKS.')
    parser.add_argument('--param_range', type=float, nargs=2, default=None,
                        help='MULTIPLIER range [min, max] for a param sub-task, drawn '
                             'log-uniformly. Sub-task 0 is always 1.0x, the stock body. '
                             'Default: env-specific.')
    parser.add_argument('--noise_range', type=float, default=1.0,
                        help='Scale for observation noise (task definition)')
    parser.add_argument('--no_gifs', action='store_true',
                        help='Skip rendering rollout GIFs (10 rollouts plus matplotlib '
                             'per task switch, used by no metric).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='continual_neuroevolution_gymnax')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--traj_steps', type=int, default=50,
                        help='Sub-sampled trajectory length fed to AURORA')
    add_diversity_args(parser)
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    
    env_name = args.env
    seed = args.seed + args.trial  # Different seed per trial
    trial = args.trial
    
    # Get env-specific config
    cfg = ENV_CONFIGS[env_name]
    num_generations = args.num_generations or cfg["num_generations"]
    pop_size = args.pop_size or cfg["pop_size"]
    hidden_dims = tuple(args.hidden_dims) if args.hidden_dims else cfg["hidden_dims"]
    episode_length = cfg["episode_length"]
    num_evals = args.num_evals or cfg["num_evals"]
    report_episodes = args.report_episodes
    algo_cfg = resolve_algo_config(args.algo, env_name, cfg)
    sigma = args.sigma or algo_cfg["sigma"]
    learning_rate = args.learning_rate or algo_cfg["learning_rate"]
    # 'es' stays the slug for OpenES: it is the name every existing run
    # directory, metrics.yaml and figure already uses.
    algo_slug = 'es' if args.algo == 'openes' else args.algo
    task_interval = args.task_interval
    task_period = args.task_period
    noise_range = args.noise_range
    task_type = args.task_type
    # ONE table for every trainer (source/utils/task_sequence.py).
    param_cfg = GYMNAX_PHYSICS_TASKS.get(env_name, {'param': None, 'mult_range': None})
    param_name = args.param_name or param_cfg['param']
    param_range = args.param_range if args.param_range is not None else param_cfg['mult_range']

    output_dir = args.output_dir or f"projects/gymnax/{algo_slug}_{env_name}_continual/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    gifs_dir = os.path.join(output_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, "train.log")
    sys.stdout = Tee(log_file)
    
    print("=" * 60)
    print(f"{args.algo.upper()} on {env_name} (CONTINUAL)")
    print("=" * 60)
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}")
    print(f"  Task interval: {task_interval} gens")
    print(f"  Task type: {task_type}")
    if task_type == 'noise':
        print(f"  Noise range: {noise_range}")
    else:
        print(f"  Param: {param_name}, multiplier range: {param_range}")
    print(f"  Sigma: {sigma}, LR: {learning_rate}")
    print(f"  Num evals: {num_evals}")
    
    key = jax.random.key(seed)
    
    # Create environment
    env, env_params = make_gymnax_env(env_name)
    env_params = env_params.replace(max_steps_in_episode=episode_length)
    # The stock body, kept aside. A param sub-task is a multiplier applied to
    # THIS, not to whatever the previous sub-task left behind.
    base_env_params = env_params
    
    # Get dimensions
    key, reset_key = jax.random.split(key)
    obs, _ = env.reset(reset_key, env_params)
    obs_dim = obs.shape[-1]
    action_dim = env.action_space(env_params).n
    # AFTER the spaces are read, so they come off the raw gymnax env with bare
    # params. FlipEnv reverses the action inside `step` under a FlippedParams
    # whose `flip` is a traced scalar, so a switching run still compiles once;
    # `__getattr__` forwards everything else to the wrapped env.
    if task_type == 'actions':
        env = wrap_actions(env)
    
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")

    # Pre-generate the deterministic per-trial sub-task sequence. This is
    # seeded from the trial index alone, NOT from the training key, so an ES
    # run and a GA/DNS/PPO run at the same trial face exactly the same
    # sub-tasks and are comparable run for run.
    num_tasks = num_generations // task_interval
    task_param_values = None
    if task_type == 'param':
        # Multipliers on the stock body, sub-task 0 = 1.0x, cycled by
        # --task_period exactly as the noise sequence is. The observation is
        # untouched, so the noise vector stays zero throughout.
        task_param_values = physics_mult_sequence(
            trial, num_tasks, param_range, task_period)
        task_noise_vectors = [jnp.zeros((obs_dim,))] * num_tasks
        print(f"  Pre-generated {num_tasks} {param_name} multipliers "
              f"(task 0 = 1.0x): {[round(m, 4) for m in task_param_values]}")
    elif task_type == 'actions':
        # Alternating, NOT drawn from the trial seed -- see action_flip_sequence.
        task_flips = action_flip_sequence(num_tasks, task_period,
                                          env_name=env_name, trial=trial)
        task_noise_vectors = [jnp.zeros((obs_dim,))] * num_tasks
        print(f"  Pre-generated {num_tasks} action-reversal flags "
              f"(task 0 = stock order): {task_flips}")
    else:
        task_noise_vectors = make_task_noise_vectors(trial, num_tasks, obs_dim, noise_range)
        task_noise_vectors = cycle_task_sequence(task_noise_vectors, task_period)
        print(f"  Pre-generated {num_tasks} noise vectors (task 0 = zero noise)")

    # Create policy
    key, init_key = jax.random.split(key)
    policy, param_template = create_policy_network(init_key, obs_dim, action_dim, hidden_dims)
    flat_params = get_flat_params(param_template)
    num_params = flat_params.shape[0]
    print(f"  Network: {hidden_dims}, {num_params} params")
    
    # Create scoring function
    scoring_fn = make_scoring_fn(env, env_params, policy, param_template,
                                 episode_length, num_evals)
    # The REPORTED evaluation, compiled once for a batch of TWO -- the
    # centroid and the generation's elite, scored together so the pair costs
    # one trace. `report_episodes` rather than `num_evals` on purpose; see
    # --report_episodes.
    report_scoring_fn = make_scoring_fn(env, env_params, policy,
                                        param_template, episode_length,
                                        report_episodes)

    # Second scoring function, with the observer descriptors; called only on
    # the generations the tracker asks for.
    track_diversity = bool(getattr(args, 'track_diversity', False))
    behaviour_cfg = BehaviourConfig(
        env_name=env_name, num_actions=int(action_dim),
        occupancy_bins=args.occupancy_bins) if track_diversity else None
    scoring_fn_bd = make_scoring_fn(
        env, env_params, policy, param_template, episode_length, num_evals,
        behaviour_cfg, args.traj_steps) if track_diversity else None

    tracker = None
    if track_diversity:
        def apply_flat(flat_params, obs_batch):
            return policy.apply(unflatten_params(flat_params, param_template), obs_batch)

        tracker = PopulationDiversityTracker(
            env_name=env_name, obs_dim=obs_dim, num_actions=int(action_dim),
            traj_steps=args.traj_steps, num_generations=num_generations,
            interval=args.diversity_interval, occupancy_bins=args.occupancy_bins,
            max_pairwise=args.diversity_max_pairwise,
            num_probe=args.num_probe_states, apply_flat=apply_flat,
            snapshots=0, seed=seed,
        )
        # Pinned per sub-task rather than spread over the run.
        tracker.snapshot_gens = continual_snapshot_gens(
            num_generations, task_interval, args.behaviour_snapshots)
        print(f"  Diversity tracking: every {args.diversity_interval} gens, "
              f"{len(tracker.snapshot_gens)} snapshots "
              f"({args.behaviour_snapshots} per sub-task)")

    # Its own random stream, so tracking cannot move the search.
    # --- Plasticity diagnostics, reported for every method (core/plasticity.py).
    # Same probe batch, churn and dormancy definitions as the RL trainers and the
    # stationary NE blocks, so every cell of the table is the same measurement.
    # Private RNG stream; an observer, never fed back into the search.
    plast = plasticity.NEPlasticityTracker(
        apply_flat=lambda f, o: policy.apply(unflatten_params(f, param_template), o),
        unflatten=lambda f: unflatten_params(f, param_template),
        num_hidden=len(hidden_dims),
        activation_fn=ACTIVATIONS[GYMNAX_POLICY_ACTIVATION],
        criterion=redo_mod.criterion_for_activation(GYMNAX_POLICY_ACTIVATION),
        num_probe=args.num_probe_states,
    )
    plasticity_key = random.key(seed + 3_000_000)
    diversity_key = random.key(seed + 1_000_000)
    probe_key = random.key(seed + 2_000_000)
    
    # Initialize wandb
    config = {
        'algo': args.algo, 'std_lr': args.std_lr,
        'env': env_name, 'num_generations': num_generations,
        'pop_size': pop_size, 'seed': seed, 'trial': trial,
        'sigma': sigma, 'learning_rate': learning_rate,
        'num_evals': num_evals, 'report_episodes': report_episodes,
        'hidden_dims': hidden_dims,
        'task_interval': task_interval, 'task_period': task_period,
        'noise_range': noise_range, 'task_type': task_type,
        'param_name': param_name,
        'param_range': list(param_range) if param_range is not None else None,
        'continual': True,
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"{algo_slug}_{env_name}_continual_trial{trial}", reinit=True)
    
    # Setup ES with multi-GPU sharding
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(np.array(devices), axis_names=('p',))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())
    parallel_sharding = NamedSharding(mesh, PartitionSpec('p'))
    
    if pop_size % num_devices != 0:
        old_pop_size = pop_size
        pop_size = (pop_size // num_devices + 1) * num_devices
        print(f"  Warning: Adjusted pop_size from {old_pop_size} to {pop_size}")
    
    es = build_strategy(args.algo, pop_size, num_params, sigma, learning_rate,
                        std_lr=args.std_lr)
    es_params = jax.device_put(es.default_params, replicate_sharding)
    
    key, init_key = random.split(key)
    es_state = es.init(init_key, jnp.zeros(num_params), es_params)
    es_state = jax.device_put(es_state, replicate_sharding)
    
    @jax.jit
    def jit_ask(key, state, params):
        population, new_state = es.ask(key, state, params)
        population = jax.device_put(population, parallel_sharding)
        return population, new_state
    
    @jax.jit
    def jit_tell(key, population, fitness, state, params):
        return es.tell(key, population, fitness, state, params)
    
    # Task 0: the stock body under no offset -- identical to non-continual
    # under either task type.
    noise_vector = task_noise_vectors[0]
    current_task = 0
    # The current sub-task's body, passed to the jitted scoring functions.
    # None under `noise`, where the body never changes and the scoring
    # functions' own closure is the stock one -- so a noise run makes exactly
    # the calls it made before this argument existed.
    ep_arg = None
    if task_type == 'param':
        env_params = apply_physics(env_name, base_env_params, param_name,
                                   task_param_values[0])
        ep_arg = env_params
        print(f"\n  Task 0 {param_name}: {task_param_values[0]:.4f}x (stock body)")
    elif task_type == 'actions':
        env_params = FlippedParams(base_env_params, jnp.float32(task_flips[0]))
        ep_arg = env_params
        print(f"\n  Task 0 action order: "
              f"{'REVERSED' if task_flips[0] else 'stock'}")
    else:
        print(f"\n  Task 0: no noise (baseline)")
        print(f"  Noise magnitude: 0.0000")

    # Warmup JIT
    print("\nJIT compiling...")
    key, warmup_key, warmup_ask_key = random.split(key, 3)
    warmup_pop, _ = jit_ask(warmup_ask_key, es_state, es_params)
    _ = scoring_fn(warmup_pop, warmup_key, noise_vector, ep_arg)[:2]
    print("  JIT compilation complete!")
    
    # Training loop
    best_overall_fitness = -float('inf')
    start_time = time.time()
    render_fn = get_render_fn(env_name)
    training_metrics = []  # Per-generation history, saved to training_metrics.json

    # One agent per sub-task for source/studies/evaluate_continual.py: the best
    # member of the sub-task's final generation, and the ES distribution mean.
    # Note both are read at the *task switch*, not from es_state.best_solution,
    # which is the best over the whole run and so belongs to whichever sub-task
    # happened to be easiest.
    ckpt_finalgen = []
    ckpt_incumbent = []
    # The network `centroid_fitness` scores, saved so the plasticity figure can
    # describe the individual the centroid lineplot draws. For ES/NES that IS
    # the distribution mean, so this duplicates `ckpt_incumbent` -- kept as its
    # own key anyway, because for the GA and DNS the two are different objects
    # and the readers must not have to know which arm they are looking at.
    ckpt_centroid = []
    population_host = None
    mean_fitness_host = None

    # Per-sub-task evaluation, written to metrics.yaml at the end of the run.
    # Same three quantities, the same 10-episode convention and the same
    # threshold set as the GA and DNS continual trainers, so the three are one
    # measurement. Without this the run saves its agents and never scores them,
    # which left ES absent from every figure keyed off metrics.yaml even where
    # its runs had finished.
    solved_threshold = SOLVED_THRESHOLDS.get(env_name)
    task_start_gen = 0
    # Set at a task switch, consumed by that task's first record.
    pending_zero_shot = None
    task_gens_to_threshold = None   # gens into the sub-task when it first cleared
    all_metrics = []                # one dict per sub-task

    def evaluate_params(flat_params, eval_noise, key_in, episodes=10):
        """Mean/std return of one genome over `episodes` fresh episodes.

        The same rollout the GIFs use, so the number reported for a sub-task is
        the one a rendered episode of it would show. Deliberately not the
        fitness the search saw: that is a max over the population and one
        `--num_evals` draw, and neither is a property of the agent alone.
        """
        rewards = []
        k = key_in
        for _ in range(episodes):
            k, ek = random.split(k)
            _, r = rollout_for_gif(env, env_params, policy, flat_params,
                                   param_template, episode_length, ek, eval_noise)
            rewards.append(r)
        return float(np.mean(rewards)), float(np.std(rewards)), k

    # Sub-task 0's zero-shot return: the initial distribution's mean, before any
    # search. Its counterpart for every later sub-task is measured at the switch
    # below, on the agent carried out of the sub-task before it.
    zt_params = np.asarray(jax.device_get(es_state.mean)).copy()
    task_zt_mean, task_zt_std, key = evaluate_params(zt_params, noise_vector, key)
    print(f"  Task 0 zero-shot: {task_zt_mean:.2f} +/- {task_zt_std:.2f}")

    print(f"\nStarting continual training...")

    for gen in range(num_generations):
        # Check if we need to switch task
        if gen > 0 and gen % task_interval == 0:
            # The two agents evaluate_continual.py scores for this sub-task.
            task_best_params = population_host[int(np.argmax(mean_fitness_host))].copy()
            ckpt_finalgen.append(task_best_params)
            ckpt_incumbent.append(np.asarray(jax.device_get(es_state.mean)).copy())
            ckpt_centroid.append(np.asarray(jax.device_get(es_state.mean)).copy())

            # Save 10 GIFs for THIS TASK before switching
            if not args.no_gifs:
                task_gif_dir = os.path.join(gifs_dir, f"task{current_task}")
                os.makedirs(task_gif_dir, exist_ok=True)
                try:
                    for gif_idx in range(10):
                        key, gif_key = random.split(key)
                        obs_list, total_reward = rollout_for_gif(
                            env, env_params, policy, task_best_params, param_template,
                            episode_length, gif_key, noise_vector
                        )

                        frames = []
                        fig, ax = None, None
                        for idx, obs in enumerate(obs_list[::2]):
                            step = idx * 2  # Actual step in episode
                            frame, fig, ax = render_fn(obs, fig, ax, step=step)
                            frames.append(frame)
                        plt.close(fig)

                        gif_path = os.path.join(task_gif_dir, f"task{current_task}_rollout_{gif_idx:02d}_reward{total_reward:.0f}.gif")
                        imageio.mimsave(gif_path, frames, fps=30, loop=0)

                    print(f"  Saved 10 GIFs for task {current_task} in {task_gif_dir}")
                except Exception as e:
                    print(f"  Warning: Failed to save GIFs: {e}")

            # What the sub-task ended at, scored on the sub-task it was
            # trained on -- this is the S column.
            task_eval_mean, task_eval_std, key = evaluate_params(
                task_best_params, noise_vector, key)
            print(f"  Task {current_task} eval: {task_eval_mean:.2f} "
                  f"+/- {task_eval_std:.2f}")
            wandb.summary[f"task_{current_task}_eval_mean"] = task_eval_mean

            all_metrics.append({
                'task_idx': current_task,
                'eval_mean': task_eval_mean,
                'eval_std': task_eval_std,
                'zero_shot_eval_mean': task_zt_mean,
                'zero_shot_eval_std': task_zt_std,
                'gens_to_threshold': task_gens_to_threshold,
            })

            # Switch to new task
            current_task += 1
            noise_vector = task_noise_vectors[current_task]
            print(f"\n>>> Task {current_task} started at gen {gen}")
            if task_type == 'param':
                # Rebound, not mutated in place: `evaluate_params` and the GIF
                # rollouts read this name at call time, so they follow the
                # switch, and `ep_arg` carries it into the jitted scoring
                # functions.
                env_params = apply_physics(env_name, base_env_params, param_name,
                                           task_param_values[current_task])
                ep_arg = env_params
                print(f"  {param_name}: {task_param_values[current_task]:.4f}x")
            elif task_type == 'actions':
                # Rebound for the same reason as the param branch above.
                flip = task_flips[current_task]
                env_params = FlippedParams(base_env_params, jnp.float32(flip))
                ep_arg = env_params
                print(f"  action order: {'REVERSED' if flip else 'stock'}")
            else:
                print(f"  Full noise: {jax.device_get(noise_vector)}")
                print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")

            # Zero-shot on the new sub-task: the agent just carried across the
            # boundary, before a single generation of search on it. Recorded
            # against the sub-task it is measured on, which is why it is stored
            # when that sub-task closes rather than here.
            task_zt_mean, task_zt_std, key = evaluate_params(
                task_best_params, noise_vector, key)
            print(f"  Task {current_task} zero-shot: {task_zt_mean:.2f} "
                  f"+/- {task_zt_std:.2f}")
            # Also carried onto the first per-generation record of the new
            # task, under the field name every suite uses. Before this it lived
            # only in wandb.summary and in the per-task metrics, so the
            # per-generation curves had no zero-shot column at all -- which is
            # how the brax and mujoco analyses came to substitute the first
            # logged generation for it. See source/metrics/zero_shot.py.
            pending_zero_shot = (task_zt_mean, current_task - 1)
            wandb.summary[f"task_{current_task}_zero_shot_mean"] = task_zt_mean

            task_start_gen = gen
            task_gens_to_threshold = None

        key, ask_key, eval_key, tell_key = random.split(key, 4)

        population, es_state = jit_ask(ask_key, es_state, es_params)
        # gen 0 always computes descriptors, so the plasticity probe batch can
        # be frozen from the initial population's visited states.
        measure_now = (tracker is not None and tracker.needs(gen)) or gen == 0
        fitness, mean_fitness, extras = (
            scoring_fn_bd if measure_now else scoring_fn)(
                population, eval_key, noise_vector, ep_arg)

        mean_fitness_host = jax.device_get(mean_fitness)
        population_host = jax.device_get(population)

        # evosax minimizes, so negate fitness for maximization (use single-trial for selection)
        es_state, _ = jit_tell(tell_key, population, -fitness, es_state, es_params)

        # THE REPORTED PAIR, scored AFTER this generation's update, on FRESH
        # keys, over `report_episodes` episodes -- the same protocol and the
        # same episode count as the RL trainer's `mean_reward`, and
        # deliberately not the search's own numbers.
        #
        #   centroid_fitness    the score of the mean of the population's
        #                       WEIGHTS. Not `mean_fitness`, which is the mean
        #                       of their FITNESSES and a different quantity.
        #   elite_eval_fitness  the genome `best_fitness` names, re-scored out
        #                       of sample. `best_fitness` is a MAX over
        #                       pop_size noisy means, so the winner's curse
        #                       inflates it -- by +5 for the GA and +53 for DNS
        #                       on the sigma=1.0 tree.
        #
        # One call on a batch of two, so the pair costs one trace and
        # 2 x report_episodes rollouts against pop x num_evals.
        elite_idx = int(np.argmax(mean_fitness_host))
        key, report_key = random.split(key)
        report_scores = np.asarray(jax.device_get(report_scoring_fn(
            jnp.stack([es_state.mean, population[elite_idx]]),
            report_key, noise_vector, ep_arg)[1]))
        centroid_fitness = float(report_scores[0])
        elite_eval_fitness = float(report_scores[1])

        # Track using mean-of-10 fitness
        gen_best = float(np.max(mean_fitness_host))
        gen_mean = float(np.mean(mean_fitness_host))

        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best

        # Time-to-solve within the sub-task, which is what SU is a ratio of.
        # Counted in generations since the sub-task started, and only the first
        # crossing counts -- fitness is noisy and recrossing the threshold later
        # is not a second solve.
        if solved_threshold is not None and task_gens_to_threshold is None:
            if gen_best >= solved_threshold:
                task_gens_to_threshold = gen - task_start_gen
                print(f"  Task {current_task} reached threshold "
                      f"{solved_threshold} at gen {gen} "
                      f"(gens_in_task={task_gens_to_threshold})")

        log_dict = {
            "generation": gen, "best_fitness": gen_best,
            "mean_fitness": gen_mean, "best_overall": best_overall_fitness,
            "centroid_fitness": centroid_fitness,
            "elite_eval_fitness": elite_eval_fitness,
            "task": current_task,
            "noise_magnitude": float(jnp.linalg.norm(noise_vector)),
        }
        if task_type == 'param':
            log_dict[f"{param_name}_mult"] = float(task_param_values[current_task])
        elif task_type == 'actions':
            log_dict['action_flip'] = int(task_flips[current_task])
        if pending_zero_shot is not None:
            attach_zero_shot(log_dict, *pending_zero_shot)
            pending_zero_shot = None

        # Plasticity: elite churn across generations, within-population churn,
        # and dormancy. See source/metrics/plasticity.py.
        if gen == 0 and measure_now and not plast.started():
            plast.start(plasticity_key, extras["observations"])
        _pop_host = jax.device_get(population)
        _elite_flat = _pop_host[int(np.argmax(np.asarray(mean_fitness_host)))]
        # incumbent=: neither OpenES nor NES has a parent/offspring relation --
        # the population is mean + sigma*eps, pure exploration noise, and no
        # sample descends from another. Its ONE update is mean_t -> mean_t+1, which is the
        # closest NE analogue to a gradient step, so that is what ne_churn
        # measures here. See NEPlasticityTracker.update.
        # `centroid=` gives every column above a `ne_centroid_*` twin measured
        # on the distribution mean -- the network `centroid_fitness` scores and
        # the one the centroid lineplot draws. The elite columns describe a
        # SAMPLED offspring, `mean + sigma*eps`, so at sigma=1.0 two consecutive
        # elites differ by a fresh noise draw and their churn is not the search
        # moving. Both are logged; the figure picks by `--agent`.
        log_dict.update(plast.update(
            jax.random.fold_in(plasticity_key, gen), _pop_host, _elite_flat,
            incumbent=jax.device_get(es_state.mean),
            centroid=jax.device_get(es_state.mean)))

        # Weight statistics of the searched parameters themselves -- the third
        # plasticity signal CLAUDE.md asks for, alongside dormancy and churn,
        # and the one that catches norms growing without bound across the
        # sub-task sequence. Every generation; see train_GA_gymnax.py. The ES
        # mean is reported separately because it, not any sampled member, is
        # what the search carries across a task boundary.
        log_dict.update(population_weight_stats(_pop_host))
        log_dict.update(weight_stats(jax.device_get(es_state.mean),
                                     prefix="mean_weight"))

        diversity = None
        if measure_now and tracker is not None:
            if gen == 0:
                tracker.start(probe_key, extras["observations"])
            diversity = tracker.update(
                random.fold_in(diversity_key, gen), gen, population,
                extras["observations"], extras["behaviour"], fitness)
            if diversity:
                log_dict.update(diversity)

        wandb.log(log_dict)
        training_metrics.append({**log_dict, 'elapsed_time': time.time() - start_time})

        if gen % args.log_interval == 0 or gen == num_generations - 1:
            print(f"Gen {gen:4d} | Task {current_task} | Best: {gen_best:8.2f} | Mean: {gen_mean:8.2f} | Overall: {best_overall_fitness:8.2f}")

    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Best overall: {best_overall_fitness:.2f}")
    print(f"  Total tasks: {current_task + 1}")
    
    # The final sub-task's agents, closing the two lists.
    final_best_params = population_host[int(np.argmax(mean_fitness_host))].copy()
    ckpt_finalgen.append(final_best_params)
    ckpt_incumbent.append(np.asarray(jax.device_get(es_state.mean)).copy())
    ckpt_centroid.append(np.asarray(jax.device_get(es_state.mean)).copy())

    # Best solution from ES state, for the run-level checkpoint below. This is
    # the best over the *whole run*, so it belongs to whichever sub-task went
    # best; the per-sub-task agents are the ones in checkpoints.npz.
    best_params = jax.device_get(es._unravel_solution(es_state.best_solution))

    # Save 10 GIFs for final task
    if not args.no_gifs:
        task_gif_dir = os.path.join(gifs_dir, f"task{current_task}")
        os.makedirs(task_gif_dir, exist_ok=True)
        try:
            for gif_idx in range(10):
                key, gif_key = random.split(key)
                obs_list, total_reward = rollout_for_gif(
                    env, env_params, policy, final_best_params, param_template,
                    episode_length, gif_key, noise_vector
                )

                frames = []
                fig, ax = None, None
                for idx, obs in enumerate(obs_list[::2]):
                    step = idx * 2  # Actual step in episode
                    frame, fig, ax = render_fn(obs, fig, ax, step=step)
                    frames.append(frame)
                plt.close(fig)

                gif_path = os.path.join(task_gif_dir, f"task{current_task}_rollout_{gif_idx:02d}_reward{total_reward:.0f}.gif")
                imageio.mimsave(gif_path, frames, fps=30, loop=0)

            print(f"Saved 10 GIFs for final task {current_task} in {task_gif_dir}")
        except Exception as e:
            print(f"Warning: Failed to save final GIFs: {e}")

    # The final sub-task never reaches a switch, so it is closed here on the
    # same terms as the others.
    task_eval_mean, task_eval_std, key = evaluate_params(
        final_best_params, noise_vector, key)
    print(f"  Task {current_task} eval: {task_eval_mean:.2f} +/- {task_eval_std:.2f}")
    all_metrics.append({
        'task_idx': current_task,
        'eval_mean': task_eval_mean,
        'eval_std': task_eval_std,
        'zero_shot_eval_mean': task_zt_mean,
        'zero_shot_eval_std': task_zt_std,
        'gens_to_threshold': task_gens_to_threshold,
    })

    # metrics.yaml: the training-time score of the run, in the shape every
    # continual figure reads. success_rate here is against solved_threshold and
    # over 10 episodes per sub-task; the numbers to *report* are the ones
    # evaluate_continual.py computes from checkpoints.npz at a controlled
    # episode count, and the two are not the same measurement.
    num_solved = sum(1 for m in all_metrics
                     if solved_threshold is not None
                     and m['eval_mean'] >= solved_threshold)
    zt_values = [m['zero_shot_eval_mean'] for m in all_metrics
                 if m['zero_shot_eval_mean'] is not None]
    summary_metrics = {
        'method': algo_slug,
        'env': env_name,
        'task_type': task_type,
        'num_tasks': len(all_metrics),
        'solved_threshold': solved_threshold,
        'success_rate': (num_solved / len(all_metrics)) if all_metrics else 0.0,
        'num_solved': num_solved,
        'zero_shot_transfer_mean': float(np.mean(zt_values)) if zt_values else None,
        'zero_shot_transfer_std': float(np.std(zt_values)) if zt_values else None,
        'per_task': all_metrics,
    }
    metrics_path = os.path.join(output_dir, "metrics.yaml")
    with open(metrics_path, 'w') as f:
        yaml.dump(summary_metrics, f, default_flow_style=False)
    print(f"Saved metrics: {metrics_path}")

    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"{algo_slug}_{env_name}_continual_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': np.array(best_params),
            'param_template': param_template,
            'best_fitness': best_overall_fitness,
            'config': config,
            'final_task': current_task,
        }, f)
    print(f"Saved: {ckpt_path}")

    # Per-generation history, for the learning-curve figure.
    save_training_metrics(output_dir, training_metrics)
    if tracker is not None:
        tracker.save(output_dir)
        print(f"Saved {len(tracker.snapshot_gens)} population snapshots to "
              f"behaviour_snapshots.npz")
    print(f"Saved {len(training_metrics)} generations to training_metrics.json")

    # Artifacts for post-hoc evaluation.
    save_eval_artifacts(
        output_dir, method=algo_slug, env=env_name, trial=trial, seed=seed,
        pop_size=pop_size, finalgen=ckpt_finalgen, incumbent=ckpt_incumbent,
        centroid=ckpt_centroid,
        noise_vectors=task_noise_vectors, hidden_dims=hidden_dims,
        episode_length=episode_length, num_generations=num_generations,
        task_interval=task_interval, num_evals=num_evals,
        noise_range=noise_range, elapsed_seconds=total_time,
        gen_best_trace=[r['best_fitness'] for r in training_metrics],
        gen_mean_trace=[r['mean_fitness'] for r in training_metrics],
        per_task=all_metrics, config=config,
        task_type=task_type,
        param_name=param_name if task_type == 'param' else None,
        param_mults=task_param_values if task_type == 'param' else None,
        action_flips=task_flips if task_type == 'actions' else None,
    )
    print(f"Saved {len(ckpt_finalgen)} sub-task agents to checkpoints.npz; "
          "score them with source/studies/evaluate_continual.py")

    wandb.finish()
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
