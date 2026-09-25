"""
Train DNS (Dominated Novelty Search) on Gymnax environments (CONTINUAL).

Task changes every 200 generations by either:
  - Adding observation noise (task_type=noise)
  - Varying an environment parameter (task_type=param):
      CartPole: gravity [0.98, 98.0] (default 9.8, factor of 10)
      MountainCar: gravity [0.000833, 0.0075] (default 0.0025, factor of 3)
      Acrobot: link_length_1 [0.5, 2.0] (default 1.0)

Custom DNS implementation matching the mujoco version.

Behaviour descriptors follow the DNS paper: either hand-designed descriptors or
*unsupervised* descriptors learned online with AURORA. Here we default to the
unsupervised variant (--descriptor aurora), i.e. the descriptor of an
individual is the latent code of an LSTM auto-encoder trained on the
observation trajectories of the current population (see source/metrics/aurora.py).
The encoder is also retrained right after every task switch, since the
observation distribution changes with the task.

Supports CartPole-v1, Acrobot-v1, MountainCar-v0.

Usage:
    python train_DNS_gymnax_continual.py --env CartPole-v1 --gpus 0
    python train_DNS_gymnax_continual.py --env CartPole-v1 --task_type param --gpus 0
"""

import argparse
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
from source.utils.runtime import Tee, _get_gpu_arg

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
import pickle
import wandb
import numpy as np
import imageio
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from source.metrics.aurora import (
    AuroraDescriptors,
    aurora_training_schedule,
    episode_relative_indices,
)
from source.utils.run_artifacts import save_eval_artifacts, save_training_metrics
from source.utils.task_sequence import (
    GYMNAX_PHYSICS_TASKS, action_flip_sequence, cycle_task_sequence,
    physics_mult_sequence)
from source.envs.gymnax_classic import (
    DEEPSEA_ENV_NAMES, DEEPSEA_SIZES, FlippedParams, apply_physics,
    make_gymnax_env, wrap_actions)

# Behavioural-diversity tracking. Its descriptors are deliberately kept
# separate from DNS's *selection* descriptor, so the observer never scores
# the method on the quantity it is optimising.
from source.metrics.behaviour_descriptors import (
    BehaviourConfig,
    average_over_evals,
    rollout_behaviour,
)
from source.metrics.zero_shot import attach as attach_zero_shot
from source.metrics.behaviour_tracking import (
    PopulationDiversityTracker,
    add_diversity_args,
    continual_snapshot_gens,
    summarise,
)

# Solved thresholds for the training-time Speed-Up (SU) diagnostic. Read from
# the shared table rather than restated here -- this trainer had -90/-120 while
# the GA and RL ones had -70/-110, so the diagnostic was not comparable across
# methods. These do not decide any reported number; those are recomputed
# post-hoc by source/studies/evaluate_continual.py.
from source.metrics.evaluation_metrics import THRESHOLD_SETS

SOLVED_THRESHOLDS = THRESHOLD_SETS['rebuttal']


# ============================================================================
# Logging Helper
# ============================================================================


# ============================================================================
# Environment Configs
# ============================================================================

ENV_CONFIGS = {
    "CartPole-v1": {
        "num_generations": 2000,  # 10 tasks x 200 gens
        "pop_size": 512,
        "batch_size": 256,
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 10,
        "k": 3,
        "iso_sigma": 0.05,
        "line_sigma": 0.5,
        "task_interval": 200,
        "num_tasks": 10,
    },
    "Acrobot-v1": {
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
        "task_interval": 200,
        "num_tasks": 10,
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
        "num_evals": 10,
        "batch_size": 256,
        "k": 3,
        "iso_sigma": 0.05,
        "line_sigma": 0.5,
        "task_interval": 200,
        "num_tasks": 10,
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


def make_scoring_fn(env, policy, param_template, episode_length, env_name,
                    num_evals=10, traj_steps=10, behaviour_cfg=None):
    """Create scoring function that returns fitness and observation trajectories.

    Evaluates each individual with num_evals trials:
    - Returns fitness from FIRST trial only (for selection)
    - Returns the FIRST trial's observation trajectory (for the descriptor)
    - Returns mean fitness across all trials (for logging/tracking)

    The trajectory is sub-sampled at traj_steps timesteps spread over the part
    of the episode that actually ran, not over the episode cap -- see
    `episode_relative_indices`. Spreading over the cap puts most samples after
    termination on early-terminating tasks, so the trajectory ends up
    describing termination rather than behaviour.

    env_params is passed as an argument (not closure-captured) so it can change per task.
    """
    num_traj_steps = min(traj_steps, episode_length)

    def evaluate_single(flat_params, eval_key, noise_vector, env_params):
        """Evaluate a single individual and return fitness + observation trajectory."""
        params = unflatten_params(flat_params, param_template)
        obs, state = env.reset(eval_key, env_params)

        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
            # Add noise to observation for continual learning
            noisy_obs = obs + noise_vector
            logits = policy.apply(params, noisy_obs)
            action = jnp.argmax(logits)  # Deterministic for evaluation

            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)

            total_reward = total_reward + reward * (1.0 - done_flag)
            valid = 1.0 - done_flag
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
            # Hold the last valid observation after termination (the env has
            # auto-reset, so next_obs would come from a new episode).
            next_obs = jnp.where(done_flag > 0, obs, next_obs)

            per_step = ((obs, valid) if behaviour_cfg is None
                        else (obs, valid, noisy_obs, action))
            return (next_obs, next_state, total_reward, done_flag, key), per_step

        key = eval_key
        (_, _, total_reward, _, _), per_step = jax.lax.scan(
            step_fn, (obs, state, 0.0, 0.0, key), None, length=episode_length
        )
        if behaviour_cfg is None:
            all_obs, valid = per_step
        else:
            all_obs, valid, all_noisy, all_actions = per_step

        # Sample within the episode that actually happened, so no sampled step
        # is post-termination padding.
        obs_traj = all_obs[episode_relative_indices(valid, num_traj_steps)]

        if behaviour_cfg is None:
            return total_reward, obs_traj
        # Observer descriptors are built from the noisy observations the policy
        # acted on, which is what defines the sub-task.
        behaviour = rollout_behaviour(all_noisy, all_actions, valid, behaviour_cfg)
        return total_reward, obs_traj, behaviour

    vmapped_eval = jax.vmap(evaluate_single, in_axes=(0, 0, None, None))

    @jax.jit
    def scoring_fn(flat_genotypes, key, noise_vector, env_params):
        """Returns (fitness_for_selection, observations, mean_fitness_for_logging)."""
        pop_size = flat_genotypes.shape[0]

        # Always evaluate with num_evals trials per individual
        all_keys = random.split(key, pop_size * num_evals)
        flat_params_repeated = jnp.repeat(flat_genotypes, num_evals, axis=0)
        if behaviour_cfg is None:
            all_fitnesses, all_observations = vmapped_eval(
                flat_params_repeated, all_keys, noise_vector, env_params)
            all_behaviour = None
        else:
            all_fitnesses, all_observations, all_behaviour = vmapped_eval(
                flat_params_repeated, all_keys, noise_vector, env_params)
        all_fitnesses = all_fitnesses.reshape(pop_size, num_evals)
        all_observations = all_observations.reshape(pop_size, num_evals, num_traj_steps, -1)

        # Fitness for selection: use FIRST trial only
        fitnesses = all_fitnesses[:, 0]
        observations = all_observations[:, 0]  # Use first trial trajectories too

        # Mean fitness for logging/tracking
        mean_fitnesses = jnp.mean(all_fitnesses, axis=1)

        if behaviour_cfg is None:
            return fitnesses, observations, mean_fitnesses
        return fitnesses, observations, mean_fitnesses, average_over_evals(
            all_behaviour, pop_size, num_evals)

    return scoring_fn


def rollout_for_gif(env, env_params, policy, flat_params, param_template, episode_length, key, noise_vector=None):
    """Rollout for GIF generation."""
    if noise_vector is None:
        noise_vector = jnp.zeros(env.observation_space(env_params).shape)
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
# Diversity Metrics
# ============================================================================




# ============================================================================
# Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='DNS on Gymnax (Continual)')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0',
                                 *DEEPSEA_ENV_NAMES])
    parser.add_argument('--num_generations', type=int, default=None)
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--iso_sigma', type=float, default=None)
    parser.add_argument('--line_sigma', type=float, default=None)
    parser.add_argument('--variation', type=str, default=ISOLINE,
                        choices=list(VARIATIONS),
                        help="how offspring are bred. 'isoline' is DNS as "
                             "published (Iso+LineDD, a two-parent operator). "
                             "'gaussian' swaps in the GA's single-parent "
                             "mutation and changes NOTHING else -- selection "
                             "is still dominated novelty -- so that a DNS-over-"
                             "GA gap can be attributed to novelty selection "
                             "rather than to recombination. Pair it with "
                             "--mutation_std and with --variation isoline on "
                             "the GA side; see "
                             "source/algorithms/ne/variation.py.")
    parser.add_argument('--mutation_std', type=float, default=0.5,
                        help="the gaussian operator's width; ignored under "
                             "--variation isoline. The default is the GA's "
                             'gymnax reference value, so the crossed arm '
                             'mutates at the scale the GA was tuned at.')
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
    add_diversity_args(parser)
    parser.add_argument('--traj_steps', type=int, default=50,
                        help='Number of evenly spaced timesteps kept per episode as AE input')
    parser.add_argument('--aurora_latent_dim', type=int, default=6,
                        help='Dimensionality of the learned descriptor space')
    parser.add_argument('--aurora_train_ratio', type=int, default=8,
                        help='Auto-encoder is retrained at generations train_ratio*cumsum(1,2,3,...)')
    parser.add_argument('--aurora_lr', type=float, default=1e-3)
    parser.add_argument('--aurora_batch_size', type=int, default=128)
    parser.add_argument('--aurora_retrain_on_task_switch', type=int, default=0,
                        help='Refit the AURORA encoder at each sub-task boundary. '
                             'OFF by default, and the default is the point: every '
                             'other method here is never told a switch happened, so '
                             'refitting the descriptor space exactly when the '
                             'observation distribution changes hands DNS alone that '
                             'signal. Refitting is the better descriptor -- the space '
                             'is learned from the observations and the switch is what '
                             'changes them -- which is why it was the default until '
                             '2026-07-30, and why it is kept as an option rather than '
                             'removed. Matches --aurora_retrain_on_task_switch in '
                             'source/studies/mujoco/train_DNS_cheetah_continual.py.')
    parser.add_argument('--refresh_population', type=int, default=1,
                        help='Re-score the repertoire EVERY generation, in the '
                             'same batch as the offspring, so dominated novelty '
                             'only ever compares numbers measured on the current '
                             'sub-task. --pop_size is then the evaluation budget: '
                             '--repertoire_ratio of it is the repertoire and the '
                             'rest are offspring (512 / 0.5 -> 256 + 256 = the '
                             "GA's 512 evaluations a generation). The boundary-"
                             'free cure for a stale repertoire, as GASearcher / '
                             'DNSSearcher(refresh=True) in the generalists study. '
                             '0 is the published algorithm: a member keeps the '
                             'fitness and descriptor it entered with, 512 members '
                             'and --batch_size offspring. Default ON since '
                             '2026-09-16; every run made before carries it off.')
    parser.add_argument('--repertoire_ratio', type=float, default=0.5)
    parser.add_argument('--reeval_population', type=int, default=0,
                        help='Re-score the whole population AT each sub-task '
                             'boundary. OFF by default: it tells DNS alone where '
                             'the switch is, which CLAUDE.md rule (d) forbids, '
                             'and under --refresh_population it is redundant. It '
                             'was unconditional until 2026-09-16, so every '
                             'continual DNS run made before then had it on.')
    parser.add_argument('--task_interval', type=int, default=200)
    parser.add_argument('--task_period', type=int, default=0,
                        help='Revisit sub-tasks: after this many distinct ones '
                             'the sequence cycles, so a 20-sub-task run at '
                             'period 10 sees each sub-task twice. 0 (default) '
                             'means every sub-task is new, which is every run '
                             'made before this flag existed.')
    parser.add_argument('--noise_range', type=float, default=1.0,
                        help='Scale for observation noise (task definition)')
    parser.add_argument('--task_type', type=str, default='noise',
                        choices=['noise', 'param', 'actions'],
                        help='Type of task variation: noise (obs noise) or param (vary env parameter)')
    parser.add_argument('--param_name', type=str, default=None,
                        help='Which physics group a param sub-task rescales -- a key of '
                             'PHYSICS_PARAMS[env] in source/envs/gymnax_classic.py. '
                             'Default: the env entry in GYMNAX_PHYSICS_TASKS.')
    parser.add_argument('--param_range', type=float, nargs=2, default=None,
                        help='MULTIPLIER range [min, max] for a param sub-task, drawn '
                             'log-uniformly. NOT an absolute parameter range: sub-task 0 '
                             'is always 1.0x, the stock body. Default: env-specific.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='continual_neuroevolution_gymnax')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_population', action='store_true',
                        help='Store the full population in each task checkpoint. '
                             'Required to compute the Recovery (R) metric, which '
                             'resumes evolution from the end-of-training state.')
    parser.add_argument('--no_gifs', action='store_true',
                        help='Skip per-task GIF rendering (10 GIFs x 10 tasks is '
                             'CPU-bound and dominates wall clock).')
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
    refresh = bool(args.refresh_population)
    reeval_population = bool(args.reeval_population)
    if refresh:
        # `pop_size` is the evaluation budget; the repertoire rides along in
        # every scored batch, so it and the offspring share it.
        if args.batch_size is not None:
            raise SystemExit('--batch_size is the --refresh_population 0 knob; '
                             'under refresh the split is --repertoire_ratio')
        repertoire_size = max(1, int(pop_size * args.repertoire_ratio))
        batch_size = pop_size - repertoire_size
        if batch_size < 1:
            raise SystemExit('--refresh_population needs --repertoire_ratio < 1')
    else:
        repertoire_size = pop_size
        batch_size = args.batch_size if args.batch_size is not None else max(1, pop_size // 2)
        batch_size = min(batch_size, pop_size)
    hidden_dims = tuple(args.hidden_dims) if args.hidden_dims else cfg["hidden_dims"]
    episode_length = cfg["episode_length"]
    num_evals = args.num_evals or cfg["num_evals"]
    report_episodes = args.report_episodes
    k = args.k or cfg["k"]
    iso_sigma = args.iso_sigma or cfg["iso_sigma"]
    line_sigma = args.line_sigma or cfg["line_sigma"]
    # The operator, resolved once. `resolve_params` rejects a knob the chosen
    # operator does not have, so a run cannot claim a width it never applied.
    variation = args.variation
    variation_params = (
        resolve_params(ISOLINE, iso_sigma=iso_sigma, line_sigma=line_sigma)
        if variation == ISOLINE else
        resolve_params(GAUSSIAN, sigma=args.mutation_std))
    task_interval = args.task_interval
    task_period = args.task_period
    noise_range = args.noise_range
    task_type = args.task_type
    
    # ONE table for every trainer (source/utils/task_sequence.py). Each of GA,
    # DNS and RL used to carry its own PARAM_CONFIGS literal and they did not
    # agree -- CartPole's range was [0.98, 98.0] here and [0.098, 198.0] in the
    # RL trainer -- so the NE and RL arms of one compute-matched comparison
    # drew different sub-task sequences from the same trial-seeded key.
    param_cfg = GYMNAX_PHYSICS_TASKS.get(env_name, {'param': None, 'mult_range': None})
    param_name = args.param_name or param_cfg['param']
    param_range = args.param_range if args.param_range is not None else param_cfg['mult_range']
    
    output_dir = args.output_dir or f"projects/gymnax/dns_{env_name}_continual_{task_type}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    gifs_dir = os.path.join(output_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, "train.log")
    sys.stdout = Tee(log_file)
    
    print("=" * 60)
    print(f"DNS on {env_name} (CONTINUAL)")
    print("=" * 60)
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}, Batch: {batch_size}")
    if refresh:
        print(f"  Repertoire refreshed every generation: {batch_size} offspring "
              f"+ {repertoire_size} re-scored members = "
              f"{batch_size + repertoire_size} evaluations/generation")
    print(f"  Re-evaluate at sub-task boundary: {reeval_population}")
    print(f"  Task interval: {task_interval} gens")
    if task_type == 'noise':
        print(f"  Task type: noise, range: {noise_range}")
    elif task_type == 'param':
        print(f"  Task type: param ({param_name}), range: {param_range}")
    print(f"  k (novelty neighbors): {k}")
    print(f"  iso_sigma: {iso_sigma}, line_sigma: {line_sigma}")
    print(f"  Num evals: {num_evals}")
    print(f"  Descriptor: {args.descriptor}")
    if args.descriptor == 'aurora':
        print(f"    Trajectory steps: {args.traj_steps}, latent dim: {args.aurora_latent_dim}")
        print(f"    AE lr: {args.aurora_lr}, train ratio: {args.aurora_train_ratio}")
        print(f"    Retrain on task switch: {bool(args.aurora_retrain_on_task_switch)}")

    key = jax.random.key(seed)
    
    # Create environment
    env, env_params = make_gymnax_env(env_name)
    env_params = env_params.replace(max_steps_in_episode=episode_length)
    # The stock body, kept aside. A param sub-task is a multiplier applied to
    # THIS, not to whatever the previous sub-task left behind, or the
    # rescalings would compound down the sequence.
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
    
    # Pre-generate deterministic task sequence (same across methods for same trial)
    num_tasks = num_generations // task_interval
    task_rng = jax.random.key(trial * 7919)  # Separate RNG, deterministic per trial
    if task_type == 'noise':
        task_noise_vectors = [jnp.zeros((obs_dim,))]  # Task 0 = no noise (identical to non-continual)
        for t in range(1, num_tasks):
            task_rng, noise_key = random.split(task_rng)
            nv = random.normal(noise_key, (obs_dim,)) * noise_range
            task_noise_vectors.append(nv)
        task_noise_vectors = cycle_task_sequence(task_noise_vectors, task_period)
        print(f"  Pre-generated {num_tasks} noise vectors (task 0 = zero noise)"
              + (f", cycling with period {task_period}" if task_period else ""))
    elif task_type == 'param':
        # Multipliers on the stock body, sub-task 0 = 1.0x, and CYCLED by
        # --task_period exactly as the noise sequence above is. The param
        # branch used to skip the cycling: a 20-sub-task run at period 10 gave
        # the noise arms each sub-task twice and the param arms twenty
        # distinct ones, so the param arms revisited nothing and could not
        # measure forgetting at all.
        task_param_values = physics_mult_sequence(
            trial, num_tasks, param_range, task_period)
        # The observation is untouched under this task type, so the offset
        # sequence is zeros -- carried explicitly rather than left undefined
        # so the saved artifacts have the same shape under both families and
        # evaluate_continual.py needs no branch to load them.
        task_noise_vectors = [jnp.zeros((obs_dim,))] * num_tasks
        print(f"  Pre-generated {num_tasks} {param_name} multipliers "
              f"(task 0 = 1.0x): {[round(m, 4) for m in task_param_values]}")
    elif task_type == 'actions':
        # Alternating, NOT drawn from the trial seed -- a two-state regime must
        # not be sampled or trials get different numbers of real switches. See
        # action_flip_sequence. The observation and the body are both untouched
        # here, so the offset sequence is zeros, carried explicitly so the saved
        # artifacts have the same shape under all three families.
        task_flips = action_flip_sequence(num_tasks, task_period,
                                          env_name=env_name, trial=trial)
        task_noise_vectors = [jnp.zeros((obs_dim,))] * num_tasks
        print(f"  Pre-generated {num_tasks} action-reversal flags "
              f"(task 0 = stock order): {task_flips}")
    
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create policy
    key, init_key = jax.random.split(key)
    policy, param_template = create_policy_network(init_key, obs_dim, action_dim, hidden_dims)
    flat_params = get_flat_params(param_template)
    num_params = flat_params.shape[0]
    print(f"  Network: {hidden_dims}, {num_params} params")
    
    # Create scoring function
    scoring_fn = make_scoring_fn(env, policy, param_template, episode_length, env_name,
                                 num_evals, args.traj_steps)
    # The REPORTED evaluation, compiled once for a batch of TWO -- the
    # centroid and the generation's elite, scored together so the pair costs
    # one trace. `report_episodes` rather than `num_evals` on purpose; see
    # --report_episodes.
    report_scoring_fn = make_scoring_fn(env, policy, param_template,
                                        episode_length, env_name,
                                        report_episodes, args.traj_steps)

    # Second scoring function, with the observer descriptors. Called only on the
    # generations the tracker asks for: collecting them costs an extra pass.
    track_diversity = bool(getattr(args, 'track_diversity', False))
    behaviour_cfg = BehaviourConfig(
        env_name=env_name, num_actions=int(action_dim),
        occupancy_bins=args.occupancy_bins) if track_diversity else None
    scoring_fn_bd = make_scoring_fn(
        env, policy, param_template, episode_length, env_name, num_evals,
        args.traj_steps, behaviour_cfg) if track_diversity else None

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
    plasticity_key = jax.random.key(seed + 3_000_000)
    diversity_key = jax.random.key(seed + 1_000_000)
    probe_key = jax.random.key(seed + 2_000_000)

    # Behaviour descriptors: unsupervised (AURORA) by default
    use_aurora = args.descriptor == 'aurora'
    retrain_on_task_switch = bool(args.aurora_retrain_on_task_switch)
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
        'refresh_population': refresh, 'repertoire_size': repertoire_size,
        'reeval_population': reeval_population,
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
        'hidden_dims': hidden_dims, 'task_interval': task_interval,
        'task_period': task_period,
        'task_type': task_type, 'noise_range': noise_range,
        'param_name': param_name, 'param_range': param_range, 'continual': True,
        'descriptor': args.descriptor, 'traj_steps': args.traj_steps,
        'aurora_latent_dim': args.aurora_latent_dim,
        'aurora_train_ratio': args.aurora_train_ratio,
        'aurora_lr': args.aurora_lr,
        'aurora_retrain_on_task_switch': retrain_on_task_switch,
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"dns_{env_name}_continual_{task_type}_pop{pop_size}_trial{trial}", reinit=True)
    
    # Initialize population
    key, pop_key = jax.random.split(key)
    population = random.normal(pop_key, (repertoire_size, num_params)) * 0.1
    
    # Generate initial noise vector (task 0)
    noise_vector = jnp.zeros((obs_dim,))
    current_task = 0
    
    if task_type == 'noise':
        noise_vector = task_noise_vectors[0]
    elif task_type == 'actions':
        env_params = FlippedParams(base_env_params, jnp.float32(task_flips[0]))
        print(f"\n  Task 0 action order: "
              f"{'REVERSED' if task_flips[0] else 'stock'}")
    elif task_type == 'param':
        param_val = task_param_values[0]
        env_params = apply_physics(env_name, base_env_params, param_name, param_val)
    
    # Initial evaluation
    print(f"\nEvaluating initial population...")
    key, eval_key = jax.random.split(key)
    fitnesses, observations, mean_fitnesses = scoring_fn(population, eval_key, noise_vector, env_params)

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

    print(f"  Initial best fitness (mean-of-10): {float(jnp.max(mean_fitnesses)):.2f}")
    if task_type == 'noise':
        print(f"  Task 0 noise vector: {jax.device_get(noise_vector)}")
        print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")
    elif task_type == 'param':
        print(f"  Task 0 {param_name}: {task_param_values[0]:.4f}x")
    
    # Training loop
    best_overall_fitness = -float('inf')
    best_params = None
    # Track per-task best separately (for GIF evaluation)
    task_best_fitness = -float('inf')
    task_best_params = None
    start_time = time.time()
    render_fn = get_render_fn(env_name)
    
    # Metrics tracking
    solved_threshold = SOLVED_THRESHOLDS.get(env_name)
    task_start_gen = 0
    # Set at a task switch, consumed by that task's first record.
    pending_zero_shot = None
    task_gens_to_threshold = None
    all_metrics = []
    training_metrics = []  # Per-generation history, saved to training_metrics.json

    # One agent per sub-task for source/studies/evaluate_continual.py. DNS keeps
    # a repertoire rather than a distribution, so the two saved agents are the
    # two ways of picking out of it: the member with the best reported fitness
    # (mean of num_evals rollouts, the definition the other scripts use) and
    # the member with the best selection fitness (the single rollout DNS
    # actually ranks on, i.e. what the optimizer would hand back).
    ckpt_finalgen = []
    ckpt_incumbent = []
    # The network `centroid_fitness` scores: the coordinate-wise mean of the
    # repertoire. `ckpt_incumbent` is the repertoire BEST, a different network
    # -- a novelty-selected repertoire has no distribution mean, so the two
    # cannot be the same object here. Both are saved; read the centroid as
    # "has the repertoire collapsed onto one solution", never as performance.
    ckpt_centroid = []

    # Zero-shot eval for task 0: evaluate a random individual on task 0
    zt_params = jax.device_get(population[0])
    zt_rewards = []
    for eval_trial in range(10):
        key, eval_key_trial = random.split(key)
        _, trial_reward = rollout_for_gif(
            env, env_params, policy, zt_params, param_template,
            episode_length, eval_key_trial, noise_vector
        )
        zt_rewards.append(trial_reward)
    task_zt_mean = float(np.mean(zt_rewards))
    task_zt_std = float(np.std(zt_rewards))
    print(f"  Task 0 zero-shot: {task_zt_mean:.2f} +/- {task_zt_std:.2f}")
    
    print(f"\nStarting continual training...")
    
    probe_started = False
    for gen in range(num_generations):
        # Check if we need to switch task
        if gen > 0 and gen % task_interval == 0:
            # Save 10 GIFs for THIS TASK before switching
            fitness_host = jax.device_get(fitnesses)
            population_host = jax.device_get(population)
            # Use best from current population (end of task)
            mean_fitness_host_task = jax.device_get(mean_fitnesses)
            current_best_idx = int(np.argmax(mean_fitness_host_task))
            current_best_params = population_host[current_best_idx].copy()

            # The two agents evaluate_continual.py scores for this sub-task.
            ckpt_finalgen.append(current_best_params.copy())
            ckpt_incumbent.append(population_host[int(np.argmax(fitness_host))].copy())
            ckpt_centroid.append(population_host.mean(axis=0).copy())

            # Per-task evaluation with 10 trials
            print(f"\n  Task {current_task} final evaluation (10 trials)...")
            task_eval_rewards = []
            for eval_trial in range(10):
                key, eval_key_trial = random.split(key)
                _, trial_reward = rollout_for_gif(
                    env, env_params, policy, current_best_params, param_template,
                    episode_length, eval_key_trial, noise_vector
                )
                task_eval_rewards.append(trial_reward)
            task_eval_mean = float(np.mean(task_eval_rewards))
            task_eval_std = float(np.std(task_eval_rewards))
            print(f"  Task {current_task} eval: {task_eval_mean:.2f} +/- {task_eval_std:.2f}")
            wandb.summary[f"task_{current_task}_eval_mean"] = task_eval_mean
            wandb.summary[f"task_{current_task}_eval_std"] = task_eval_std
            
            # Create task subfolder and save 10 GIFs
            task_gif_dir = os.path.join(gifs_dir, f"task{current_task}")
            if not args.no_gifs:
                os.makedirs(task_gif_dir, exist_ok=True)

            try:
                for gif_idx in range(10):
                    key, gif_key = random.split(key)
                    if args.no_gifs:
                        # Still split the key, so that --no_gifs leaves the RNG
                        # stream identical to a run that renders GIFs and the
                        # two are directly comparable.
                        continue
                    obs_list, total_reward = rollout_for_gif(
                        env, env_params, policy, current_best_params, param_template,
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
                
                if not args.no_gifs:
                    print(f"  Saved 10 GIFs for task {current_task} in {task_gif_dir}")
            except Exception as e:
                print(f"  Warning: Failed to save GIFs: {e}")
            
            # Save per-task checkpoint
            task_ckpt_path = os.path.join(checkpoints_dir, f"task_{current_task}.pkl")
            ckpt_data = {
                'flat_params': np.array(current_best_params),
                'task_idx': current_task,
                'task_type': task_type,
                'generation': gen,
                'best_fitness': float(task_best_fitness),
                'eval_mean': task_eval_mean,
                'eval_std': task_eval_std,
                'zero_shot_eval_mean': task_zt_mean,
                'zero_shot_eval_std': task_zt_std,
                'gens_to_threshold': task_gens_to_threshold,
            }
            if args.save_population:
                # Needed by the Recovery (R) metric, which resumes evolution
                # from the state the agent was actually left in.
                ckpt_data['population'] = population_host.copy()
            if task_type == 'noise':
                ckpt_data['noise_vector'] = jax.device_get(noise_vector)
            elif task_type == 'param':
                ckpt_data['param_name'] = param_name
                ckpt_data['param_mult'] = float(task_param_values[current_task])
            with open(task_ckpt_path, 'wb') as f:
                pickle.dump(ckpt_data, f)
            print(f"    Saved task checkpoint: {task_ckpt_path}")
            
            # Store per-task metrics
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
            task_start_gen = gen
            task_gens_to_threshold = None
            if task_type == 'noise':
                noise_vector = task_noise_vectors[current_task]
                print(f"\n>>> Task {current_task} started at gen {gen}")
                print(f"  Full noise: {jax.device_get(noise_vector)}")
                print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")
            elif task_type == 'actions':
                flip = task_flips[current_task]
                env_params = FlippedParams(base_env_params, jnp.float32(flip))
                print(f"\n>>> Task {current_task}: action order "
                      f"{'REVERSED' if flip else 'stock'}")
            elif task_type == 'param':
                param_val = task_param_values[current_task]
                env_params = apply_physics(env_name, base_env_params,
                                           param_name, param_val)
                print(f"\n>>> Task {current_task} started at gen {gen}")
                print(f"  {param_name}: {param_val:.4f}x")
            
            # Zero-shot evaluation on new task (before training)
            zt_rewards = []
            for eval_trial in range(10):
                key, eval_key_trial = random.split(key)
                _, trial_reward = rollout_for_gif(
                    env, env_params, policy, current_best_params, param_template,
                    episode_length, eval_key_trial, noise_vector
                )
                zt_rewards.append(trial_reward)
            task_zt_mean = float(np.mean(zt_rewards))
            task_zt_std = float(np.std(zt_rewards))
            print(f"  Task {current_task} zero-shot: {task_zt_mean:.2f} +/- {task_zt_std:.2f}")
            # Also carried onto the first per-generation record of the new
            # task, under the field name every suite uses. Before this it lived
            # only in wandb.summary and in the per-task metrics, so the
            # per-generation curves had no zero-shot column at all -- which is
            # how the brax and mujoco analyses came to substitute the first
            # logged generation for it. See source/metrics/zero_shot.py.
            pending_zero_shot = (task_zt_mean, current_task - 1)
            wandb.summary[f"task_{current_task}_zero_shot_mean"] = task_zt_mean
            wandb.summary[f"task_{current_task}_zero_shot_std"] = task_zt_std
            
            # Reset task-specific best tracking for new task
            task_best_fitness = -float('inf')
            task_best_params = None
            
            # Re-evaluate population on the new task -- boundary information,
            # so behind --reeval_population (see its help).
            if reeval_population:
                key, eval_key = jax.random.split(key)
                fitnesses, observations, mean_fitnesses = scoring_fn(population, eval_key, noise_vector, env_params)

            # The observation distribution changed with the task, so retrain the
            # encoder on the new trajectories before re-encoding descriptors.
            if use_aurora and retrain_on_task_switch:
                key, ae_key = jax.random.split(key)
                aurora_state, aurora_loss = aurora.train(
                    ae_key, observations, aurora_state, iteration=gen
                )
                print(f"  Retrained AURORA encoder for task {current_task}, "
                      f"reconstruction loss: {aurora_loss:.4f}")

            descriptors = compute_descriptors(observations)
            novelties = _compute_dominated_novelty(fitnesses, descriptors, k)

        # Generate offspring
        key, var_key = jax.random.split(key)
        # return_parents: see train_DNS_gymnax.py. Pairing must be taken
        # before dns_selection merges parents and offspring.
        offspring, _parents = vary(variation, population, var_key, batch_size,
                                   variation_params, return_parents=True)
        
        key, eval_key = jax.random.split(key)
        if refresh:
            # Offspring and repertoire in ONE scored batch, on this
            # generation's sub-task and encoder, so every fitness and
            # descriptor dns_selection compares is current. No boundary is
            # consulted: after a switch the repertoire's old numbers are
            # simply overwritten on the first generation, as the GA's are.
            batch = jnp.concatenate([offspring, population], axis=0)
            b_fit, b_obs, _ = scoring_fn(batch, eval_key, noise_vector, env_params)
            b_desc = compute_descriptors(b_obs)
            nb = batch_size
            population, fitnesses, descriptors, observations, novelties = dns_selection(
                batch[nb:], b_fit[nb:], b_desc[nb:], b_obs[nb:],
                batch[:nb], b_fit[:nb], b_desc[:nb], b_obs[:nb],
                repertoire_size, k
            )
        else:
            # Evaluate offspring; parents keep their stored numbers.
            offspring_fitnesses, offspring_observations, offspring_mean_fitnesses = scoring_fn(offspring, eval_key, noise_vector, env_params)
            offspring_descriptors = compute_descriptors(offspring_observations)

            # DNS selection (uses single-trial fitness for selection)
            population, fitnesses, descriptors, observations, novelties = dns_selection(
                population, fitnesses, descriptors, observations,
                offspring, offspring_fitnesses, offspring_descriptors, offspring_observations,
                repertoire_size, k
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

        # Re-evaluate population to get mean fitness for logging
        # (since DNS selection mixed parents and offspring, we need fresh mean_fitnesses)
        # This already scores exactly the surviving population, so it is also
        # where the observer measures -- no descriptor has to be carried
        # through dns_selection.
        key, eval_key = jax.random.split(key)
        # gen 0 always computes descriptors, so the plasticity probe batch can
        # be frozen from the initial population's visited states.
        measure_now = (tracker is not None and tracker.needs(gen)) or gen == 0
        if measure_now and tracker is not None:
            obs_fitnesses, obs_observations, mean_fitnesses, obs_behaviour = \
                scoring_fn_bd(population, eval_key, noise_vector, env_params)
        else:
            _, _, mean_fitnesses = scoring_fn(population, eval_key, noise_vector, env_params)

        # Copy to host for numpy operations - use mean-of-10 for tracking
        mean_fitness_host = jax.device_get(mean_fitnesses)
        population_host = jax.device_get(population)
        
        gen_best = float(np.max(mean_fitness_host))
        gen_mean = float(np.mean(mean_fitness_host))
        best_idx = int(np.argmax(mean_fitness_host))

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
        # Expect the widest centroid gap of the three NE methods here, and
        # expect it to be a fact about novelty search rather than about
        # continual learning: DNS keeps its repertoire behaviourally DIVERSE
        # on purpose, so its members are the least likely to be permutations
        # of one solution, and averaging genuinely different networks gives a
        # network worse than any of them. Read `centroid_fitness` as "has the
        # repertoire collapsed", never as this method's performance --
        # `elite_eval_fitness` is the performance number.
        key, report_key = random.split(key)
        report_scores = np.asarray(jax.device_get(report_scoring_fn(
            jnp.stack([population.mean(axis=0), population[best_idx]]),
            report_key, noise_vector, env_params)[2]))
        centroid_fitness = float(report_scores[0])
        elite_eval_fitness = float(report_scores[1])
        
        # Update task-specific best (for GIF evaluation)
        if gen_best > task_best_fitness:
            task_best_fitness = gen_best
            task_best_params = population_host[best_idx].copy()
        
        # Track generations to threshold (SU metric)
        if solved_threshold is not None and task_gens_to_threshold is None:
            if gen_best >= solved_threshold:
                task_gens_to_threshold = gen - task_start_gen
                print(f"  Task {current_task} reached threshold {solved_threshold} at gen {gen} (gens_in_task={task_gens_to_threshold})")
        
        # Update overall best (for checkpoint)
        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best
            best_params = population_host[best_idx].copy()
        
        # Diversity metrics
        fitness_div = compute_fitness_diversity(fitnesses)
        genomic_div = compute_genomic_diversity(population)
        descriptor_div = compute_descriptor_diversity(descriptors)
        mean_novelty = float(jnp.nanmean(novelties))

        log_dict = {
            "generation": gen, "best_fitness": gen_best,
            "mean_fitness": gen_mean, "best_overall": best_overall_fitness,
            "centroid_fitness": centroid_fitness,
            "elite_eval_fitness": elite_eval_fitness,
            "fitness_diversity": fitness_div, "genomic_diversity": genomic_div,
            "descriptor_diversity": descriptor_div,
            "mean_novelty": mean_novelty, "task": current_task,
        }
        if pending_zero_shot is not None:
            attach_zero_shot(log_dict, *pending_zero_shot)
            pending_zero_shot = None
        if use_aurora:
            log_dict["aurora_loss"] = aurora_loss

        # Plasticity: elite churn across generations, within-population churn,
        # and dormancy. See source/metrics/plasticity.py.
        if gen == 0 and measure_now and not plast.started():
            plast.start(plasticity_key, obs_observations)
        _pop_host = jax.device_get(population)
        _elite_flat = _pop_host[int(np.argmax(np.asarray(mean_fitnesses)))]
        # `centroid=` measures the same columns on the repertoire mean, the
        # network `centroid_fitness` scores. The elite columns argmax fitness
        # over a repertoire selected on NOVELTY, so they hop between
        # behaviourally distinct genomes; the mean does not.
        log_dict.update(plast.update(
            jax.random.fold_in(plasticity_key, gen), _pop_host, _elite_flat,
            centroid=_pop_host.mean(axis=0),
            parents=jax.device_get(_parents), offspring=jax.device_get(offspring)))

        # Weight statistics of the searched parameters themselves -- the third
        # plasticity signal CLAUDE.md asks for, alongside dormancy and churn,
        # and the one that catches norms growing without bound across the
        # sub-task sequence. Every generation; see train_GA_gymnax.py.
        log_dict.update(population_weight_stats(_pop_host))

        diversity = None
        if measure_now:
            # Started on the first measured generation rather than on gen 0:
            # the probe batch and the observer's encoder need a population's
            # trajectories, and this hook is the first place DNS has them.
            if not probe_started:
                tracker.start(probe_key, obs_observations)
                probe_started = True
            diversity = tracker.update(
                jax.random.fold_in(diversity_key, gen), gen, population,
                obs_observations, obs_behaviour, obs_fitnesses)
            if diversity:
                log_dict.update(diversity)
        if task_type == 'noise':
            log_dict["noise_magnitude"] = float(jnp.linalg.norm(noise_vector))
        elif task_type == 'param':
            log_dict[f"{param_name}_mult"] = float(task_param_values[current_task])
        wandb.log(log_dict)
        training_metrics.append({**log_dict, 'elapsed_time': time.time() - start_time})

        if gen % args.log_interval == 0 or gen == num_generations - 1:
            print(f"Gen {gen:4d} | Task {current_task} | Best: {gen_best:8.2f} | Mean: {gen_mean:8.2f} | Overall: {best_overall_fitness:8.2f} | Nov: {mean_novelty:.3f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Best overall: {best_overall_fitness:.2f}")
    print(f"  Total tasks: {current_task + 1}")
    
    # Use best from final generation of this task
    mean_fitness_host = jax.device_get(mean_fitnesses)
    population_host = jax.device_get(population)
    final_best_idx = int(np.argmax(mean_fitness_host))
    final_best_params = population_host[final_best_idx].copy()

    # The final sub-task's agents, closing the two lists.
    ckpt_finalgen.append(final_best_params.copy())
    ckpt_incumbent.append(
        population_host[int(np.argmax(jax.device_get(fitnesses)))].copy()
    )
    ckpt_centroid.append(population_host.mean(axis=0).copy())

    # Per-task evaluation with 10 trials for final task
    print(f"\n  Task {current_task} final evaluation (10 trials)...")
    task_eval_rewards = []
    for eval_trial in range(10):
        key, eval_key_trial = random.split(key)
        _, trial_reward = rollout_for_gif(
            env, env_params, policy, final_best_params, param_template,
            episode_length, eval_key_trial, noise_vector
        )
        task_eval_rewards.append(trial_reward)
    task_eval_mean = float(np.mean(task_eval_rewards))
    task_eval_std = float(np.std(task_eval_rewards))
    print(f"  Task {current_task} eval: {task_eval_mean:.2f} +/- {task_eval_std:.2f}")
    wandb.summary[f"task_{current_task}_eval_mean"] = task_eval_mean
    wandb.summary[f"task_{current_task}_eval_std"] = task_eval_std
    
    task_gif_dir = os.path.join(gifs_dir, f"task{current_task}")
    if not args.no_gifs:
        os.makedirs(task_gif_dir, exist_ok=True)

    try:
        for gif_idx in range(10):
            key, gif_key = random.split(key)
            if args.no_gifs:
                continue
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
        
        if not args.no_gifs:
            print(f"Saved 10 GIFs for final task {current_task} in {task_gif_dir}")
    except Exception as e:
        print(f"Warning: Failed to save final GIFs: {e}")
    
    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"dns_{env_name}_continual_best.pkl")
    best_ckpt_data = {
        'flat_params': np.array(best_params) if best_params is not None else None,
        'param_template': param_template,
        'best_fitness': best_overall_fitness,
        'config': config,
        'final_task': current_task,
    }
    if use_aurora:
        # Keep the final encoder so descriptors can be recomputed offline
        best_ckpt_data['aurora_state'] = jax.device_get(aurora_state)
    with open(ckpt_path, 'wb') as f:
        pickle.dump(best_ckpt_data, f)
    print(f"Saved: {ckpt_path}")
    
    # Save final task checkpoint
    task_ckpt_path = os.path.join(checkpoints_dir, f"task_{current_task}.pkl")
    final_ckpt_data = {
        'flat_params': np.array(final_best_params),
        'task_idx': current_task,
        'task_type': task_type,
        'generation': num_generations,
        'best_fitness': float(task_best_fitness),
        'eval_mean': task_eval_mean,
        'eval_std': task_eval_std,
        'zero_shot_eval_mean': task_zt_mean,
        'zero_shot_eval_std': task_zt_std,
        'gens_to_threshold': task_gens_to_threshold,
    }
    if args.save_population:
        final_ckpt_data['population'] = jax.device_get(population)
    if task_type == 'noise':
        final_ckpt_data['noise_vector'] = jax.device_get(noise_vector)
    elif task_type == 'param':
        final_ckpt_data['param_name'] = param_name
        final_ckpt_data['param_mult'] = float(task_param_values[current_task])
    with open(task_ckpt_path, 'wb') as f:
        pickle.dump(final_ckpt_data, f)
    print(f"Saved final task checkpoint: {task_ckpt_path}")
    
    # Store final task metrics
    all_metrics.append({
        'task_idx': current_task,
        'eval_mean': task_eval_mean,
        'eval_std': task_eval_std,
        'zero_shot_eval_mean': task_zt_mean,
        'zero_shot_eval_std': task_zt_std,
        'gens_to_threshold': task_gens_to_threshold,
    })
    
    # Compute aggregate metrics and save to YAML
    num_tasks = len(all_metrics)
    num_solved = sum(1 for m in all_metrics if solved_threshold is not None and m['eval_mean'] >= solved_threshold)
    success_rate = num_solved / num_tasks if num_tasks > 0 else 0.0
    zt_values = [m['zero_shot_eval_mean'] for m in all_metrics]
    
    summary_metrics = {
        # The operator is part of the arm's identity: `dns_gaussian` is this
        # trainer's dominated-novelty selection over gaussian-mutated
        # offspring, and must not be averaged in with `dns`.
        'method': 'dns' if variation == ISOLINE else 'dns_gaussian',
        'env': env_name,
        'task_type': task_type,
        'num_tasks': num_tasks,
        'solved_threshold': solved_threshold,
        'success_rate': success_rate,
        'num_solved': num_solved,
        'zero_shot_transfer_mean': float(np.mean(zt_values)),
        'zero_shot_transfer_std': float(np.std(zt_values)),
        'per_task': all_metrics,
    }
    metrics_path = os.path.join(output_dir, "metrics.yaml")
    with open(metrics_path, 'w') as f:
        yaml.dump(summary_metrics, f, default_flow_style=False)
    print(f"Saved metrics: {metrics_path}")

    # Per-generation history, for the learning-curve figure.
    save_training_metrics(output_dir, training_metrics)
    if tracker is not None:
        tracker.save(output_dir)
        print(f"Saved {len(tracker.snapshot_gens)} population snapshots to "
              f"behaviour_snapshots.npz")
    print(f"Saved {len(training_metrics)} generations to training_metrics.json")

    # Artifacts for post-hoc evaluation. The success_rate in metrics.yaml above
    # comes from the small evaluation done during training; the numbers to
    # report are the ones evaluate_continual.py computes from these agents at a
    # controlled episode count.
    #
    # EMITTED UNDER BOTH TASK TYPES since 2026-09-08. This used to be gated on
    # `task_type == 'noise'` because the evaluator could only rebuild an
    # observation offset, so a param run saved its agents nowhere, wrote no
    # results.json, and was invisible to verify_runs.py and make_lineplot.py's
    # budget check as well as to the evaluator. The multiplier sequence is now
    # saved alongside the (zero) offsets and the evaluator rebuilds the body
    # with apply_physics.
    if True:
        save_eval_artifacts(
            output_dir,
            method='dns' if variation == ISOLINE else 'dns_gaussian',
            env=env_name, trial=trial, seed=seed,
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
