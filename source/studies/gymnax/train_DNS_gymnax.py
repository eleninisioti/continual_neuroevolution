"""
Train DNS (Dominated Novelty Search) on Gymnax environments (non-continual).

Custom DNS implementation matching the mujoco version.

Behaviour descriptors follow the DNS paper: either hand-designed descriptors or
*unsupervised* descriptors learned online with AURORA. Here we default to the
unsupervised variant (--descriptor aurora), i.e. the descriptor of an
individual is the latent code of an LSTM auto-encoder trained on the
observation trajectories of the current population (see source/metrics/aurora.py).
The hand-designed alternative (--descriptor handcrafted) picks a couple of
observation dimensions at the end of the episode; for these gymnax tasks there
is no established descriptor, which is why the learned one is the default.

Supports CartPole-v1, Acrobot-v1, MountainCar-v0.
All are discrete action space environments.

Usage:
    python train_DNS_gymnax.py --env CartPole-v1 --gpus 0
    python train_DNS_gymnax.py --env Acrobot-v1 --gpus 0
    python train_DNS_gymnax.py --env MountainCar-v0 --gpus 0
"""

import argparse
import json
import os
import sys
import time

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
from source.algorithms.ne.dns import _compute_dominated_novelty, compute_descriptor_diversity, compute_fitness_diversity, compute_genomic_diversity, dns_selection, handcrafted_descriptors
from source.algorithms.ne.variation import GAUSSIAN, ISOLINE, VARIATIONS, resolve_params, vary
from source.algorithms.networks import MLPPolicy, create_policy_network, get_flat_params, unflatten_params
from source.algorithms.networks import ACTIVATIONS
from source.algorithms.rl.ppo import GYMNAX_POLICY_ACTIVATION
from source.metrics import plasticity
from source.metrics.weight_stats import population_weight_stats
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
import flax.linen as nn
import gymnax
from source.envs.gymnax_classic import make_gymnax_env, wrap_actions
import pickle
import wandb
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from source.metrics.aurora import (
    AuroraDescriptors,
    aurora_training_schedule,
    episode_relative_indices,
    subsample_indices,
)
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
# Environment Configs
# ============================================================================

ENV_CONFIGS = {
    "CartPole-v1": {
        "num_generations": 500,
        "pop_size": 512,
        "batch_size": 256,
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 10,
        "k": 3,
        "iso_sigma": 0.05,
        "line_sigma": 0.5,
    },
    "Acrobot-v1": {
        "num_generations": 1000,
        "pop_size": 512,
        "batch_size": 256,
        # (16, 16), not (32, 32): the policy is pinned across every
        # method compared on this env (source/algorithms/networks.py
        # POLICY_ARCH). It read (32, 32) here while every run on disk
        # was launched with an explicit --hidden_dims 16 16.
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 10,
        "k": 3,
        "iso_sigma": 0.05,
        "line_sigma": 0.5,
    },
    "MountainCar-v0": {
        "num_generations": 2000,
        "pop_size": 512,
        "batch_size": 256,
        # (16, 16), not (32, 32): the policy is pinned across every
        # method compared on this env (source/algorithms/networks.py
        # POLICY_ARCH). It read (32, 32) here while every run on disk
        # was launched with an explicit --hidden_dims 16 16.
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 10,
        "k": 3,
        "iso_sigma": 0.05,
        "line_sigma": 0.5,
    },
}


# ============================================================================
# Policy Network (Discrete Actions)
# ============================================================================





# ============================================================================
# DNS Algorithm
# ============================================================================




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


# ============================================================================
# Scoring Function
# ============================================================================


def make_scoring_fn(env, env_params, policy, param_template, episode_length, env_name,
                    num_evals=10, traj_steps=10, behaviour_cfg=None):
    """Create scoring function that returns fitness and observation trajectories.

    Evaluates each individual with num_evals trials:
    - Returns fitness from FIRST trial only (for selection)
    - Returns the FIRST trial's observation trajectory (for the descriptor)
    - Returns mean fitness across all trials (for logging/tracking)

    The trajectory is sub-sampled at traj_steps evenly spaced timesteps and
    frozen at the last valid observation once the episode terminates (gymnax
    auto-resets on `done`, so later observations belong to a fresh episode).

    With `behaviour_cfg` set it also returns the hand-designed / occupancy /
    action-frequency descriptors averaged over the trials, for the diversity
    tracking; DNS's own selection still uses only `descriptors`.
    """
    traj_indices = subsample_indices(episode_length, traj_steps)
    num_traj_steps = int(traj_indices.shape[0])
    track = behaviour_cfg is not None

    def evaluate_single(flat_params, eval_key):
        """Evaluate a single individual and return fitness + observation trajectory."""
        params = unflatten_params(flat_params, param_template)
        obs, state = env.reset(eval_key, env_params)

        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
            logits = policy.apply(params, obs)
            action = jnp.argmax(logits)  # Deterministic for evaluation

            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)

            total_reward = total_reward + reward * (1.0 - done_flag)
            valid = 1.0 - done_flag
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
            # Hold the last valid observation after termination (the env has
            # auto-reset, so next_obs would come from a new episode).
            next_obs = jnp.where(done_flag > 0, obs, next_obs)

            per_step = (obs, action, valid) if track else (obs, valid)
            return (next_obs, next_state, total_reward, done_flag, key), per_step

        key = eval_key
        (_, _, total_reward, _, _), per_step = jax.lax.scan(
            step_fn, (obs, state, 0.0, 0.0, key), None, length=episode_length
        )

        if not track:
            all_obs, valid = per_step
            return (total_reward,
                    all_obs[episode_relative_indices(valid, num_traj_steps)])

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
        """Returns (fitness_for_selection, observations, mean_fitness, behaviour).

        `behaviour` is None unless the scoring function was built with a
        BehaviourConfig.
        """
        pop_size = flat_genotypes.shape[0]

        # Always evaluate with num_evals trials per individual
        all_keys = random.split(key, pop_size * num_evals)
        flat_params_repeated = jnp.repeat(flat_genotypes, num_evals, axis=0)
        if track:
            all_fitnesses, all_observations, all_behaviour = vmapped_eval(
                flat_params_repeated, all_keys)
        else:
            all_fitnesses, all_observations = vmapped_eval(flat_params_repeated, all_keys)
        all_fitnesses = all_fitnesses.reshape(pop_size, num_evals)
        all_observations = all_observations.reshape(pop_size, num_evals, num_traj_steps, -1)

        # Fitness for selection: use FIRST trial only
        fitnesses = all_fitnesses[:, 0]
        observations = all_observations[:, 0]

        # Mean fitness for logging/tracking
        mean_fitnesses = jnp.mean(all_fitnesses, axis=1)

        behaviour = (average_over_evals(all_behaviour, pop_size, num_evals)
                     if track else None)
        return fitnesses, observations, mean_fitnesses, behaviour

    return scoring_fn


def make_eval_best_fn(env, env_params, policy, param_template, episode_length, num_eval_trials=10):
    """Create function to evaluate a single individual with multiple trials for logging."""
    
    def evaluate_single(flat_params, eval_key):
        params = unflatten_params(flat_params, param_template)
        obs, state = env.reset(eval_key, env_params)
        
        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
            logits = policy.apply(params, obs)
            action = jnp.argmax(logits)
            
            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)
            
            total_reward = total_reward + reward * (1.0 - done_flag)
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
            
            return (next_obs, next_state, total_reward, done_flag, key), None
        
        (_, _, total_reward, _, _), _ = jax.lax.scan(
            step_fn, (obs, state, 0.0, 0.0, eval_key), None, length=episode_length
        )
        return total_reward
    
    vmapped_eval = jax.vmap(evaluate_single, in_axes=(None, 0))
    
    @jax.jit
    def eval_best_fn(flat_params, key):
        """Evaluate single individual with multiple trials, return mean fitness."""
        keys = random.split(key, num_eval_trials)
        fitnesses = vmapped_eval(flat_params, keys)
        return jnp.mean(fitnesses)
    
    return eval_best_fn


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
        total_reward += float(reward)
        obs_list.append(np.array(obs))
        
        if bool(done):
            if verbose:
                print(f"  Episode ended at step {step}, reward={total_reward:.2f}")
            break
    
    return obs_list, total_reward


# ============================================================================
# Diversity Metrics
# ============================================================================




# ============================================================================
# Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='DNS on Gymnax (Non-Continual)')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'])
    parser.add_argument('--num_generations', type=int, default=None)
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--iso_sigma', type=float, default=None)
    parser.add_argument('--line_sigma', type=float, default=None)
    parser.add_argument('--variation', type=str, default=ISOLINE,
                        choices=list(VARIATIONS),
                        help="how offspring are bred: 'isoline' is DNS as "
                             "published, 'gaussian' swaps in the GA's "
                             'single-parent mutation and changes nothing else. '
                             'The stationary half of the operator ablation; '
                             'see source/algorithms/ne/variation.py.')
    parser.add_argument('--mutation_std', type=float, default=0.5,
                        help="the gaussian operator's width; ignored under "
                             "--variation isoline")
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
    # --- Behaviour descriptors ---
    parser.add_argument('--descriptor', type=str, default='aurora',
                        choices=['aurora', 'handcrafted'],
                        help='aurora: unsupervised descriptors learned online (paper default '
                             'for tasks without established descriptors); handcrafted: a few '
                             'observation dimensions at the end of the episode')
    parser.add_argument('--traj_steps', type=int, default=50,
                        help='Number of evenly spaced timesteps kept per episode as AE input')
    parser.add_argument('--aurora_latent_dim', type=int, default=6,
                        help='Dimensionality of the learned descriptor space')
    parser.add_argument('--aurora_train_ratio', type=int, default=8,
                        help='Auto-encoder is retrained at generations train_ratio*cumsum(1,2,3,...)')
    parser.add_argument('--aurora_lr', type=float, default=1e-3)
    parser.add_argument('--aurora_batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='continual_neuroevolution_gymnax')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--dump_descriptors', type=int, default=0,
                        help='Save this many evenly-spaced snapshots of the '
                             'population descriptors/fitnesses to descriptors.npz '
                             '(0 disables). Used by the AURORA diagnostic plots.')
    parser.add_argument('--no_gifs', action='store_true',
                        help='Skip rendering the 10 evaluation GIFs. They are '
                             'CPU-bound and dominate wall clock on short runs.')
    add_diversity_args(parser)
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    
    env_name = args.env
    seed = args.seed
    trial = args.trial
    
    # Get env-specific config
    cfg = ENV_CONFIGS[env_name]
    num_generations = args.num_generations or cfg["num_generations"]
    pop_size = args.pop_size or cfg["pop_size"]
    batch_size = args.batch_size if args.batch_size is not None else max(1, pop_size // 2)
    batch_size = min(batch_size, pop_size)
    hidden_dims = tuple(args.hidden_dims) if args.hidden_dims else cfg["hidden_dims"]
    episode_length = cfg["episode_length"]
    num_evals = args.num_evals or cfg["num_evals"]
    report_episodes = args.report_episodes
    k = args.k or cfg["k"]
    iso_sigma = args.iso_sigma or cfg["iso_sigma"]
    line_sigma = args.line_sigma or cfg["line_sigma"]
    variation = args.variation
    variation_params = (
        resolve_params(ISOLINE, iso_sigma=iso_sigma, line_sigma=line_sigma)
        if variation == ISOLINE else
        resolve_params(GAUSSIAN, sigma=args.mutation_std))
    
    output_dir = args.output_dir or f"projects/gymnax/dns_{env_name}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, "train.log")
    sys.stdout = Tee(log_file)
    
    print("=" * 60)
    print(f"DNS on {env_name} (Non-Continual)")
    print("=" * 60)
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}, Batch: {batch_size}")
    print(f"  k (novelty neighbors): {k}")
    print(f"  iso_sigma: {iso_sigma}, line_sigma: {line_sigma}")
    print(f"  Num evals: {num_evals}")
    print(f"  Descriptor: {args.descriptor}")
    if args.descriptor == 'aurora':
        print(f"    Trajectory steps: {args.traj_steps}, latent dim: {args.aurora_latent_dim}")
        print(f"    AE lr: {args.aurora_lr}, train ratio: {args.aurora_train_ratio}")

    key = jax.random.key(seed)
    
    # Create environment
    env, env_params = make_gymnax_env(env_name)
    env_params = env_params.replace(max_steps_in_episode=episode_length)
    
    # Get dimensions
    key, reset_key = jax.random.split(key)
    obs, _ = env.reset(reset_key, env_params)
    obs_dim = obs.shape[-1]
    action_dim = env.action_space(env_params).n
    
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create policy
    key, init_key = jax.random.split(key)
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
    # every rollout, so it lives in a second compiled function used only for
    # the re-evaluation on generations that are actually measured; offspring
    # scoring and the final evaluations never need it.
    scoring_fn = make_scoring_fn(env, env_params, policy, param_template, episode_length,
                                 env_name, num_evals, args.traj_steps)
    # The REPORTED evaluation, compiled once for a batch of TWO -- the
    # centroid and the generation's elite, scored together so the pair costs
    # one trace. `report_episodes` rather than `num_evals` on purpose; see
    # --report_episodes. The continual trainers log the same two columns, and
    # FT subtracts one tree's from the other's, so the stationary reference
    # has to be the SAME estimator or the difference is part protocol.
    report_scoring_fn = make_scoring_fn(env, env_params, policy, param_template,
                                        episode_length, env_name,
                                        report_episodes, args.traj_steps)
    scoring_fn_bd = make_scoring_fn(
        env, env_params, policy, param_template, episode_length, env_name,
        num_evals, args.traj_steps, behaviour_cfg) if track_diversity else None

    # Behaviour descriptors: unsupervised (AURORA) by default
    use_aurora = args.descriptor == 'aurora'
    aurora, aurora_state, aurora_schedule = None, None, set()
    aurora_loss = float('nan')
    if use_aurora:
        aurora = AuroraDescriptors(
            obs_size=obs_dim, traj_steps=args.traj_steps,
            latent_dim=args.aurora_latent_dim, learning_rate=args.aurora_lr,
            batch_size=args.aurora_batch_size,
        )
        key, aurora_key = jax.random.split(key)
        aurora_state = aurora.init(aurora_key)
        aurora_schedule = aurora_training_schedule(num_generations, args.aurora_train_ratio)

    def compute_descriptors(observations):
        """Descriptors of a batch of observation trajectories, current encoder."""
        if use_aurora:
            return aurora.encode(observations, aurora_state)
        return handcrafted_descriptors(observations, env_name)

    # Initialize wandb
    config = {
        'env': env_name, 'num_generations': num_generations,
        'pop_size': pop_size, 'batch_size': batch_size, 'k': k,
        # Which operator actually bred the offspring, and at what widths.
        # `dns` and `dns_gaussian` are one trainer with one string changed, so
        # without this nothing in the run says which of them a directory holds.
        # The widths come from `variation_params`, NOT from `iso_sigma` /
        # `line_sigma` directly: those are read from the env config whatever
        # the operator is, so recording them would leave a `--variation
        # gaussian` run claiming an Iso+LineDD scale it never applied.
        'variation': variation, **variation_params,
        'seed': seed, 'trial': trial, 'num_evals': num_evals,
        'report_episodes': report_episodes,
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
        'descriptor': args.descriptor, 'traj_steps': args.traj_steps,
        'aurora_latent_dim': args.aurora_latent_dim,
        'aurora_train_ratio': args.aurora_train_ratio,
        'aurora_lr': args.aurora_lr,
        'track_diversity': track_diversity,
        'diversity_interval': args.diversity_interval,
        'occupancy_bins': args.occupancy_bins,
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"dns_{env_name}_trial{trial}", reinit=True)
    # The run says what network it searched, so
    # scripts/check_architectures.py can confirm every method compared on
    # this task used the same one. The continual trainers already wrote
    # this into results.json; the stationary ones wrote it nowhere.
    write_run_config(output_dir, config, policy_arch='gymnax')

    # Initialize population
    key, pop_key = jax.random.split(key)
    population = random.normal(pop_key, (pop_size, num_params)) * 0.1

    # Initial evaluation
    print(f"\nEvaluating initial population...")
    key, eval_key = jax.random.split(key)
    fitnesses, observations, mean_fitnesses, _ = scoring_fn(population, eval_key)

    if use_aurora:
        # First training of the encoder on the initial population's trajectories
        print(f"Training initial AURORA auto-encoder...")
        key, ae_key = jax.random.split(key)
        aurora_state, aurora_loss = aurora.train(
            ae_key, observations, aurora_state, iteration=0, verbose=True
        )
        print(f"  Initial AE reconstruction loss: {aurora_loss:.4f}")

    descriptors = compute_descriptors(observations)
    novelties = _compute_dominated_novelty(fitnesses, descriptors, k)

    print(f"  Initial best fitness (selection): {float(jnp.max(fitnesses)):.2f}")
    print(f"  Initial best fitness (mean 10): {float(jnp.max(mean_fitnesses)):.2f}")

    # Training loop
    best_overall_fitness = -float('inf')  # Tracks mean-of-10 fitness
    best_params = None
    start_time = time.time()

    # Snapshots of the descriptor space, for the offline AURORA diagnostics.
    descriptor_snapshots = {}
    if args.dump_descriptors > 0:
        snapshot_gens = set(np.linspace(
            0, num_generations - 1, args.dump_descriptors).astype(int).tolist())
    else:
        snapshot_gens = set()

    training_metrics = []  # Per-generation history, saved to training_metrics.json

    # Behavioural diversity of the population, measured several ways. DNS
    # already has an AURORA encoder when --descriptor aurora, so the tracker
    # reuses those latents instead of fitting a second encoder; with
    # --descriptor handcrafted it fits its own observer encoder.
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
            own_aurora=not use_aurora, snapshots=args.behaviour_snapshots,
            latent_dim=args.aurora_latent_dim, aurora_lr=args.aurora_lr,
            aurora_batch_size=args.aurora_batch_size,
            aurora_train_ratio=args.aurora_train_ratio, seed=seed,
        )
        # Own random stream, so tracking cannot shift the search's draws.
        tracker.start(jax.random.key(seed + 2_000_000), observations)
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
    diversity_key = jax.random.key(seed + 1_000_000)

    print(f"\nStarting training...")

    for gen in range(num_generations):
        # Generate offspring
        key, var_key = jax.random.split(key)
        # return_parents: x1, the iso base each offspring descends from. The
        # churn column needs the pairing BEFORE dns_selection merges parents
        # and offspring -- afterwards the survivors are not in correspondence
        # with anything. Same offspring either way.
        offspring, _parents = vary(variation, population, var_key, batch_size,
                                   variation_params, return_parents=True)

        # Evaluate offspring (fitness for selection, mean_fitness for tracking)
        key, eval_key = jax.random.split(key)
        offspring_fitnesses, offspring_observations, offspring_mean_fitnesses, _ = scoring_fn(offspring, eval_key)
        offspring_descriptors = compute_descriptors(offspring_observations)

        # DNS selection uses single-trial fitness
        population, fitnesses, descriptors, observations, novelties = dns_selection(
            population, fitnesses, descriptors, observations,
            offspring, offspring_fitnesses, offspring_descriptors, offspring_observations,
            pop_size, k
        )

        # Retrain the AURORA encoder on the surviving population's trajectories
        # and re-encode every stored descriptor with the new encoder.
        if use_aurora and (gen + 1) in aurora_schedule:
            key, ae_key = jax.random.split(key)
            aurora_state, aurora_loss = aurora.train(
                ae_key, observations, aurora_state, iteration=gen
            )
            descriptors = aurora.encode(observations, aurora_state)
            novelties = _compute_dominated_novelty(fitnesses, descriptors, k)
            print(f"  [gen {gen}] retrained AURORA encoder, reconstruction loss: {aurora_loss:.4f}")

        # Re-evaluate selected population to get mean fitnesses for tracking.
        # The behaviour descriptors come from these fresh rollouts, so they
        # describe exactly the individuals that survived this generation.
        key, reeval_key = jax.random.split(key)
        # gen 0 always computes descriptors: the plasticity probe batch is
        # frozen from the initial population's visited states, exactly as the
        # diversity probes are.
        measure_now = (tracker is not None and tracker.needs(gen)) or gen == 0
        _, reeval_observations, mean_fitnesses, behaviour = (
            scoring_fn_bd if measure_now else scoring_fn)(population, reeval_key)

        # Copy to host for numpy operations
        mean_fitness_host = jax.device_get(mean_fitnesses)
        population_host = jax.device_get(population)
        
        # Track using mean-of-10 fitness
        gen_best = float(np.max(mean_fitness_host))
        gen_mean = float(np.mean(mean_fitness_host))
        best_idx = int(np.argmax(mean_fitness_host))
        
        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best
            best_params = population_host[best_idx].copy()
        
        # Diversity metrics
        fitness_div = compute_fitness_diversity(fitnesses)
        genomic_div = compute_genomic_diversity(population)
        descriptor_div = compute_descriptor_diversity(descriptors)
        mean_novelty = float(jnp.nanmean(novelties))

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
        # Expect the widest centroid gap of the three NE methods here: DNS
        # keeps its repertoire behaviourally DIVERSE on purpose, so averaging
        # its members' weights averages genuinely different networks.
        key, report_key = random.split(key)
        report_scores = np.asarray(jax.device_get(report_scoring_fn(
            jnp.stack([population.mean(axis=0), population[best_idx]]),
            report_key)[2]))
        centroid_fitness = float(report_scores[0])
        elite_eval_fitness = float(report_scores[1])

        log_dict = {
            "generation": gen, "best_fitness": gen_best,
            "mean_fitness": gen_mean, "best_overall": best_overall_fitness,
            "centroid_fitness": centroid_fitness,
            "elite_eval_fitness": elite_eval_fitness,
            "fitness_diversity": fitness_div, "genomic_diversity": genomic_div,
            "descriptor_diversity": descriptor_div,
            "mean_novelty": mean_novelty,
        }
        if use_aurora:
            log_dict["aurora_loss"] = aurora_loss

        # Plasticity: churn (elite across generations, and within-population)
        # plus dormancy. Same probe batch and same definitions as the RL
        # trainers -- see source/metrics/plasticity.py.
        if gen == 0 and measure_now and not plast.started():
            plast.start(plasticity_key, reeval_observations)
        _population_host = jax.device_get(population)
        _elite_flat = _population_host[int(np.argmax(np.asarray(mean_fitnesses)))]
        plast_metrics = plast.update(
            jax.random.fold_in(plasticity_key, gen), _population_host, _elite_flat,
            parents=jax.device_get(_parents), offspring=jax.device_get(offspring))
        log_dict.update(plast_metrics)

        # Weight statistics of the searched parameters themselves -- the third
        # plasticity signal CLAUDE.md asks for, alongside dormancy and churn.
        # Every generation; see the note in train_GA_gymnax.py.
        log_dict.update(population_weight_stats(_population_host))

        diversity = None
        if measure_now and tracker is not None:
            diversity = tracker.update(
                jax.random.fold_in(diversity_key, gen), gen, population,
                reeval_observations, behaviour, fitnesses,
                aurora_descriptors=descriptors if use_aurora else None,
                aurora_loss=aurora_loss if use_aurora else None)
            if diversity:
                log_dict.update(diversity)

        wandb.log(log_dict)

        training_metrics.append({**log_dict, 'elapsed_time': time.time() - start_time})

        if gen in snapshot_gens:
            descriptor_snapshots[f"gen_{gen}_descriptors"] = jax.device_get(descriptors)
            descriptor_snapshots[f"gen_{gen}_fitnesses"] = jax.device_get(fitnesses)
            descriptor_snapshots[f"gen_{gen}_mean_fitnesses"] = mean_fitness_host
            descriptor_snapshots[f"gen_{gen}_novelties"] = jax.device_get(novelties)

        if gen % args.log_interval == 0 or gen == num_generations - 1:
            line = (f"Gen {gen:4d} | Best: {gen_best:8.2f} | Mean: {gen_mean:8.2f} "
                    f"| Overall: {best_overall_fitness:8.2f} | Nov: {mean_novelty:.3f}")
            if diversity:
                line += f" | {summarise(diversity)}"
            print(line)
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Best overall (training): {best_overall_fitness:.2f}")
    
    # Re-evaluate final population to get accurate mean fitness
    print(f"\nRe-evaluating final population (mean of 10)...")
    key, reeval_key = jax.random.split(key)
    _, _, final_mean_fitnesses, _ = scoring_fn(population, reeval_key)
    final_best_idx = int(jnp.argmax(final_mean_fitnesses))
    population_host = jax.device_get(population)
    best_params = population_host[final_best_idx].copy()
    print(f"  Re-evaluated best (mean of 10): {float(final_mean_fitnesses[final_best_idx]):.2f}")
    
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
    ckpt_path = os.path.join(output_dir, f"dns_{env_name}_best.pkl")
    ckpt_data = {
        'flat_params': np.array(best_params),
        'param_template': param_template,
        'best_fitness': best_overall_fitness,
        'final_eval_mean': final_mean,
        'final_eval_std': final_std,
        'config': config,
    }
    if use_aurora:
        # Keep the final encoder so descriptors can be recomputed offline
        ckpt_data['aurora_state'] = jax.device_get(aurora_state)
    with open(ckpt_path, 'wb') as f:
        pickle.dump(ckpt_data, f)
    print(f"Saved: {ckpt_path}")

    # Save per-generation training history
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    if descriptor_snapshots:
        desc_path = os.path.join(output_dir, "descriptors.npz")
        np.savez_compressed(
            desc_path,
            snapshot_gens=np.array(sorted(snapshot_gens)),
            **descriptor_snapshots)
        print(f"Saved {len(snapshot_gens)} descriptor snapshots: {desc_path}")

    if tracker is not None:
        snapshot_path = tracker.save(output_dir)
        if snapshot_path:
            print(f"Saved: {snapshot_path}")

    if args.no_gifs:
        print("Skipping GIF rendering (--no_gifs)")
        wandb.finish()
        print(f"\nDone! Results saved to {output_dir}")
        return

    # GIFs are rendered post-hoc from the saved checkpoint, not here -- see
    # scripts/neurips_2026_rebuttal/make_gifs.py. Rendering is host-side
    # matplotlib and has no business inside a training run.
    wandb.finish()
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
