"""
Train a distribution-based NE method on Gymnax environments (non-continual).

Two methods, one runner, chosen with `--algo`:

    openes   centered ranks + Adam. The default,
             and the arm every OpenES number in this study came from.
    nes      `source/algorithms/ne/es.py` -- standardized fitness + SGD,
             i.e. the textbook (1, lambda) search gradient.

They differ in exactly those two places and in nothing else: same network, same
population size, same evaluation budget, same metrics, same random-key schedule.
That is the point of putting them behind one flag rather than in two trainers --
a difference in the table is then a difference between the methods.

Supports CartPole-v1, Acrobot-v1, MountainCar-v0.
All are discrete action space environments.

Usage:
    python train_ES_gymnax.py --env CartPole-v1 --gpus 0
    python train_ES_gymnax.py --env Acrobot-v1 --gpus 0 --algo nes
    python train_ES_gymnax.py --env MountainCar-v0 --gpus 0
"""

import argparse
import json
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
from source.utils.runtime import Tee, _get_gpu_arg, write_run_config

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
from source.envs.gymnax_classic import make_gymnax_env, wrap_actions
import time
import pickle
import wandb
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from source.studies.gymnax.es_algorithms import build_strategy, resolve_algo_config
from source.metrics.aurora import episode_relative_indices, subsample_indices
from source.metrics.behaviour_descriptors import (
    BehaviourConfig,
    average_over_evals,
    rollout_behaviour,
)
from source.metrics.behaviour_tracking import (
    PopulationDiversityTracker,
    add_diversity_args,
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

# The old sigma=0.5 / lr=0.01 setting was the worst of the 20 (sigma, lr) pairs
# swept on every one of these three tasks, and reached the solved threshold on
# 0/5 seeds within 250 generations on all of them: sigma=0.5 perturbs the policy
# far enough that most of the population lands in the same failure mode, the
# centered ranks are then mostly ties, and lr=0.01 is too small to act on what
# signal survives. A tighter search distribution with a matching larger step
# fixes all three. Values below are per-env sweep winners over 5 seeds each
# (sigma=0.2 / lr=0.2 also solves 5/5 on all three if a single shared setting is
# wanted; it is just slower than sigma=0.1 / lr=0.1 on MountainCar).
ENV_CONFIGS = {
    # 5/5 seeds sustain 475 by generation 17 (worst 22), vs never for the old
    # setting, which plateaued around 414.
    "CartPole-v1": {
        "num_generations": 500,
        "pop_size": 512,
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 1,
        "sigma": 0.2,
        "learning_rate": 0.2,
    },
    # 5/5 seeds sustain -70 by generation 14 (worst 23), vs never for the old
    # setting, whose population mean stayed near -296.
    "Acrobot-v1": {
        "num_generations": 1000,
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
    },
    # The sparse-reward case, where the failure above is starkest: under the old
    # setting nearly every perturbation times out at -500, so the ranks really
    # are pure noise. 5/5 seeds sustain -150 by generation 24 (worst 46) and
    # settle near -93; the old setting plateaued around -348.
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
    },
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


def save_gif(frames, path, fps=30):
    """Save frames as GIF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  [DEBUG] Saving {len(frames)} frames to {path}")
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"Saved GIF: {path}")


# ============================================================================
# Scoring Function
# ============================================================================

def make_scoring_fn(env, env_params, policy, param_template, episode_length, num_evals=10,
                    behaviour_cfg=None, traj_steps=10):
    """Create JIT-compiled scoring function for gymnax.

    Evaluates each individual with num_evals trials:
    - Returns fitness from FIRST trial only (for selection)
    - Returns mean fitness across all trials (for logging/tracking)

    With `behaviour_cfg` set it additionally returns, per individual, the
    sub-sampled observation trajectory of the first trial (what AURORA encodes)
    and the behaviour descriptors averaged over all trials. Selection sees
    neither, so tracking cannot change what the run does.
    """
    traj_indices = subsample_indices(episode_length, traj_steps)
    num_traj_steps = int(traj_indices.shape[0])
    track = behaviour_cfg is not None

    def evaluate_single(flat_params, eval_key):
        params = unflatten_params(flat_params, param_template)
        reset_key, _ = random.split(eval_key)
        obs, state = env.reset(reset_key, env_params)

        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
            logits = policy.apply(params, obs)
            action = jnp.argmax(logits)  # Greedy action

            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)
            total_reward = total_reward + reward * (1.0 - done_flag)
            # Valid = this step belongs to the episode (the env auto-resets on
            # done, so later steps are a fresh episode and must be masked out).
            valid = 1.0 - done_flag
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))

            per_step = (obs, action, valid) if track else None
            return (next_obs, next_state, total_reward, done_flag, key), per_step

        (_, _, total_reward, _, _), per_step = jax.lax.scan(
            step_fn, (obs, state, 0.0, 0.0, eval_key), None, length=episode_length
        )
        if not track:
            return total_reward

        all_obs, all_actions, valid = per_step
        # Sample within the episode that actually happened, so no sampled step
        # is post-termination padding (see episode_relative_indices).
        behaviour = rollout_behaviour(all_obs, all_actions, valid, behaviour_cfg)
        return (total_reward,
                all_obs[episode_relative_indices(valid, num_traj_steps)],
                behaviour)

    vmapped_eval = jax.vmap(evaluate_single)

    @jax.jit
    def scoring_fn(flat_genotypes, key):
        """Returns (fitness_for_selection, mean_fitness_for_logging, behaviour).

        `behaviour` is None unless the scoring function was built with a
        BehaviourConfig.
        """
        pop_size = flat_genotypes.shape[0]

        # Always evaluate with num_evals trials per individual
        all_keys = random.split(key, pop_size * num_evals)
        flat_params_repeated = jnp.repeat(flat_genotypes, num_evals, axis=0)
        if track:
            all_fitnesses, all_traj, all_behaviour = vmapped_eval(
                flat_params_repeated, all_keys)
        else:
            all_fitnesses = vmapped_eval(flat_params_repeated, all_keys)
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


def rollout_for_gif(env, env_params, policy, flat_params, param_template, episode_length, key, verbose=False):
    """Rollout for evaluation/GIF generation (not JIT - needs obs extraction)."""
    params = unflatten_params(flat_params, param_template)
    obs, state = env.reset(key, env_params)
    
    obs_list = [np.array(obs)]
    total_reward = 0.0
    
    for step in range(episode_length):
        logits = policy.apply(params, obs)
        action = int(jnp.argmax(logits))
        
        key, step_key = random.split(key)
        obs, state, reward, done, _ = env.step(step_key, state, action, env_params)
        obs_list.append(np.array(obs))
        total_reward += float(reward)
        
        if bool(done):
            if verbose:
                print(f"  Episode ended at step {step}, reward={total_reward:.2f}")
            break
    
    return obs_list, total_reward


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='OpenES / NES on Gymnax (Non-Continual)')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'])
    parser.add_argument('--algo', type=str, default='openes',
                        choices=['openes', 'nes'],
                        help='openes: centered ranks + Adam. '
                             'nes: standardized fitness + SGD (search gradient).')
    parser.add_argument('--std_lr', type=float, default=0.0,
                        help='NES only: separable-NES learning rate for the '
                             'per-coordinate search width. 0 keeps sigma fixed, '
                             'which is what makes the two arms differ only in '
                             'shaping and step rule.')
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


def main():
    args = parse_args()
    
    env_name = args.env
    seed = args.seed
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

    # Directory and run names carry the method, so an NES run never lands on
    # top of the OpenES run of the same env and trial.
    algo_slug = 'es' if args.algo == 'openes' else args.algo
    output_dir = args.output_dir or f"projects/gymnax/{algo_slug}_{env_name}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, "train.log")
    sys.stdout = Tee(log_file)
    
    print(f"\n{args.algo.upper()} on {env_name} (Non-Continual)")
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}")
    print(f"  Sigma: {sigma}, LR: {learning_rate}")
    print(f"  Num evals: {num_evals}")
    
    key = random.key(seed)
    
    # Create environment
    env, env_params = make_gymnax_env(env_name)
    # Must be set explicitly: gymnax defaults to 200 steps for MountainCar (500
    # for CartPole and Acrobot), so without this ES would run shorter episodes
    # than the GA and DNS trainers on the same task.
    env_params = env_params.replace(max_steps_in_episode=episode_length)

    # Get dimensions
    key, reset_key = random.split(key)
    obs, _ = env.reset(reset_key, env_params)
    obs_dim = obs.shape[-1]
    action_dim = env.action_space(env_params).n
    
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create policy
    key, init_key = random.split(key)
    policy, param_template = create_policy_network(init_key, obs_dim, action_dim, hidden_dims)
    flat_params = get_flat_params(param_template)
    num_params = flat_params.shape[0]
    print(f"  Network: {hidden_dims}, {num_params} params")
    
    # Behavioural diversity tracking (observer only -- see
    # source/metrics/behaviour_tracking.py)
    track_diversity = bool(args.track_diversity)
    behaviour_cfg = BehaviourConfig(
        env_name=env_name, num_actions=int(action_dim),
        occupancy_bins=args.occupancy_bins) if track_diversity else None

    # Create scoring function. The descriptor pass costs an extra sweep over
    # every rollout, so it lives in a second compiled function that is only
    # called on the generations that are actually measured.
    scoring_fn = make_scoring_fn(env, env_params, policy, param_template, episode_length,
                                 num_evals)
    # The REPORTED evaluation, compiled once for a batch of TWO -- the
    # centroid and the generation's elite, scored together so the pair costs
    # one trace. `report_episodes` rather than `num_evals` on purpose; see
    # --report_episodes. The continual trainers log the same two columns, and
    # FT subtracts one tree's from the other's, so the stationary reference
    # has to be the SAME estimator or the difference is part protocol.
    report_scoring_fn = make_scoring_fn(env, env_params, policy, param_template,
                                        episode_length, report_episodes)
    scoring_fn_bd = make_scoring_fn(
        env, env_params, policy, param_template, episode_length, num_evals,
        behaviour_cfg, args.traj_steps) if track_diversity else None

    # Initialize wandb
    config = {
        'algo': args.algo, 'std_lr': args.std_lr,
        'env': env_name, 'num_generations': num_generations,
        'pop_size': pop_size, 'seed': seed, 'trial': trial,
        'sigma': sigma, 'learning_rate': learning_rate,
        'num_evals': num_evals, 'report_episodes': report_episodes,
        # episode_length is RECORDED, not left to be assumed. Every consumer
        # needs it: an NE run's budget is
        # `generations x pop_size x num_evals x episode_length` steps, and
        # without it `scripts/verify_runs.py` cannot compute-match this run
        # against PPO and `scripts/make_lineplot.py` DROPPED the method from
        # the figure without saying so -- the stationary NE arms were absent
        # from every `--phase noncontinual` plot for exactly this reason. The
        # continual trainers got away with it because save_eval_artifacts
        # writes it at the top level of results.json; these write only a
        # config.json, so the field has to be here.
        'episode_length': episode_length,
        'hidden_dims': hidden_dims,
        'track_diversity': track_diversity,
        'diversity_interval': args.diversity_interval,
        'occupancy_bins': args.occupancy_bins, 'traj_steps': args.traj_steps,
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"{algo_slug}_{env_name}_trial{trial}", reinit=True)
    # The run says what network it searched, so
    # scripts/check_architectures.py can confirm every method compared on
    # this task used the same one. The continual trainers already wrote
    # this into results.json; the stationary ones wrote it nowhere.
    write_run_config(output_dir, config, policy_arch='gymnax')
    
    # Initialize ES with multi-GPU sharding
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
    
    # JIT compile ask/tell
    @jax.jit
    def jit_ask(key, state, params):
        population, new_state = es.ask(key, state, params)
        population = jax.device_put(population, parallel_sharding)
        return population, new_state
    
    @jax.jit
    def jit_tell(key, population, fitness, state, params):
        return es.tell(key, population, fitness, state, params)
    
    # Warmup JIT
    print("\nJIT compiling...")
    key, warmup_key, warmup_ask_key = random.split(key, 3)
    warmup_pop, _ = jit_ask(warmup_ask_key, es_state, es_params)
    _ = scoring_fn(warmup_pop, warmup_key)
    print("  JIT compilation complete!")

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
            snapshots=args.behaviour_snapshots, seed=seed,
        )
    # Diversity draws from its own stream so that switching tracking on or off
    # leaves the search's random numbers -- and therefore the run -- unchanged.
    # --- Plasticity diagnostics, reported for every method (core/plasticity.py).
    # Churn and dormancy on a frozen probe batch, defined identically to the RL
    # trainers' so NE and RL sit in one table. Private RNG stream, never fed back
    # into the search -- an observer, like the diversity tracking above.
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

    # Training loop
    best_overall_fitness = -float('inf')  # Tracks mean-of-10 fitness
    start_time = time.time()
    training_metrics = []  # Per-generation history, saved to training_metrics.json

    print(f"\nStarting training...")

    for gen in range(num_generations):
        key, ask_key, eval_key, tell_key = random.split(key, 4)

        population, es_state = jit_ask(ask_key, es_state, es_params)
        # gen 0 always computes descriptors: the plasticity probe batch is
        # frozen from the initial population's visited states, exactly as the
        # diversity probes are.
        measure_now = (tracker is not None and tracker.needs(gen)) or gen == 0
        fitness, mean_fitness, extras = (
            scoring_fn_bd if measure_now else scoring_fn)(population, eval_key)
        # evosax minimizes, so negate fitness for maximization (use single-trial for selection)
        es_state, _ = jit_tell(tell_key, population, -fitness, es_state, es_params)

        mean_fitness_host = jax.device_get(mean_fitness)

        # Track using mean-of-10 fitness
        gen_best = float(np.max(mean_fitness_host))
        gen_mean = float(np.mean(mean_fitness_host))

        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best

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
            jnp.stack([es_state.mean, population[elite_idx]]), report_key)[1]))
        centroid_fitness = float(report_scores[0])
        elite_eval_fitness = float(report_scores[1])

        record = {
            'generation': gen,
            'best_fitness': gen_best,
            'mean_fitness': gen_mean,
            'best_overall': best_overall_fitness,
            'centroid_fitness': centroid_fitness,
            'elite_eval_fitness': elite_eval_fitness,
        }

        # Plasticity: churn (elite across generations, and within-population)
        # plus dormancy. Same probe batch and same definitions as the RL
        # trainers -- see source/metrics/plasticity.py.
        if gen == 0 and measure_now and not plast.started():
            plast.start(plasticity_key, extras["observations"])
        _population_host = jax.device_get(population)
        _elite_flat = _population_host[int(np.argmax(mean_fitness_host))]
        # incumbent=: neither OpenES nor NES has a parent/offspring relation --
        # the population is mean + sigma*eps, pure exploration noise, and no
        # sample descends from another. Its ONE update is mean_t -> mean_t+1, which is the
        # closest NE analogue to a gradient step, so that is what ne_churn
        # measures here. See NEPlasticityTracker.update.
        plast_metrics = plast.update(
            jax.random.fold_in(plasticity_key, gen), _population_host, _elite_flat,
            incumbent=jax.device_get(es_state.mean))
        record.update(plast_metrics)

        # Weight statistics of the searched parameters themselves -- the third
        # plasticity signal CLAUDE.md asks for, alongside dormancy and churn.
        # Every generation; see the note in train_GA_gymnax.py. The ES mean is
        # reported separately from the sampled population because it, not any
        # sampled member, is what the search actually carries forward.
        record.update(population_weight_stats(_population_host))
        record.update(weight_stats(jax.device_get(es_state.mean),
                                   prefix='mean_weight'))

        diversity = None
        if measure_now and tracker is not None:
            if gen == 0:
                tracker.start(probe_key, extras["observations"])
            diversity = tracker.update(
                jax.random.fold_in(diversity_key, gen), gen, population,
                extras["observations"], extras["behaviour"], fitness)
            if diversity:
                record.update(diversity)

        wandb.log(record)
        training_metrics.append({**record, 'elapsed_time': time.time() - start_time})

        if gen % args.log_interval == 0 or gen == num_generations - 1:
            line = (f"Gen {gen:4d} | Best: {gen_best:8.2f} | Mean: {gen_mean:8.2f} "
                    f"| Overall: {best_overall_fitness:8.2f}")
            if diversity:
                line += f" | {summarise(diversity)}"
            print(line)
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Best overall (training): {best_overall_fitness:.2f}")
    
    # Use ES mean as the best solution (ES optimizes the mean, not individual samples)
    best_params = jax.device_get(es_state.mean)
    
    # Final evaluation with 10 trials
    print(f"\nFinal evaluation (10 trials)...")
    final_eval_rewards = []
    final_eval_num_trials = 10
    for eval_trial in range(final_eval_num_trials):
        key, eval_key = random.split(key)
        _, trial_reward = rollout_for_gif(
            env, env_params, policy, best_params, param_template, episode_length, eval_key
        )
        final_eval_rewards.append(trial_reward)
        print(f"  Trial {eval_trial + 1}: {trial_reward:.2f}")
    
    final_mean = float(np.mean(final_eval_rewards))
    final_std = float(np.std(final_eval_rewards))
    final_max = float(np.max(final_eval_rewards))
    final_min = float(np.min(final_eval_rewards))
    
    print(f"\nFinal evaluation results:")
    print(f"  Mean: {final_mean:.2f} +/- {final_std:.2f}")
    print(f"  Min: {final_min:.2f}, Max: {final_max:.2f}")
    print(f"  Training best: {best_overall_fitness:.2f}")
    
    wandb.log({
        "final_eval_mean": final_mean,
        "final_eval_std": final_std,
        "final_eval_max": final_max,
        "final_eval_min": final_min,
    })
    
    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"{algo_slug}_{env_name}_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': np.array(best_params),
            'param_template': param_template,
            'best_fitness': best_overall_fitness,
            'final_eval_mean': final_mean,
            'final_eval_std': final_std,
            'config': config,
        }, f)
    print(f"Saved: {ckpt_path}")

    # Save per-generation training history
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    if tracker is not None:
        snapshot_path = tracker.save(output_dir)
        if snapshot_path:
            print(f"Saved: {snapshot_path}")

    # GIFs are rendered post-hoc from the saved checkpoint, not here -- see
    # scripts/neurips_2026_rebuttal/make_gifs.py. Rendering is host-side
    # matplotlib and has no business inside a training run.
    wandb.finish()
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
