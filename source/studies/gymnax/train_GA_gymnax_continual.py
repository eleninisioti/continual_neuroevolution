"""
Train GA on Gymnax environments (CONTINUAL).

Task changes every 200 generations by either:
  - Adding observation noise (task_type=noise)
  - Varying an environment parameter (task_type=param):
      CartPole: gravity [0.98, 98.0] (default 9.8, factor of 10)
      MountainCar: gravity [0.000833, 0.0075] (default 0.0025, factor of 3)
      Acrobot: link_length_1 [0.5, 2.0] (default 1.0)

Supports CartPole-v1, Acrobot-v1, MountainCar-v0.

Usage:
    python train_GA_gymnax_continual.py --env CartPole-v1 --gpus 0
    python train_GA_gymnax_continual.py --env CartPole-v1 --task_type param --gpus 0
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
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import flax.linen as nn
from source.algorithms.ne.ga import SimpleGA
from source.algorithms.ne.variation import (
    GAUSSIAN, ISOLINE, VARIATIONS, resolve_params,
)
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

from source.utils.run_artifacts import save_eval_artifacts, save_training_metrics
from source.utils.task_sequence import (
    GYMNAX_PHYSICS_TASKS, action_flip_sequence, cycle_task_sequence,
    physics_mult_sequence)
from source.envs.gymnax_classic import (
    DEEPSEA_ENV_NAMES, DEEPSEA_SIZES, FlippedParams, apply_physics,
    make_gymnax_env, wrap_actions)

# Solved thresholds for the training-time Speed-Up (SU) diagnostic. Read from
# the shared table rather than restated here, because the three continual
# trainers had drifted to three different sets (-70/-110 here and in the RL
# trainer, -90/-120 in the DNS one), which made even the diagnostic
# incomparable across methods.
#
# These do not decide any reported number: success rate and speed-up are
# recomputed post-hoc by source/studies/evaluate_continual.py at whichever
# threshold set is being reported.
from source.metrics.evaluation_metrics import THRESHOLD_SETS

# Behavioural-diversity tracking. Identical machinery to the noncontinual
# trainer -- an observer that never feeds selection and draws from its own
# random stream, so a run is bit-identical with it on or off.
from source.metrics.aurora import episode_relative_indices, subsample_indices
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

SOLVED_THRESHOLDS = THRESHOLD_SETS['rebuttal']


# ============================================================================
# Logging Helper
# ============================================================================


# ============================================================================
# Policy Network (Discrete Actions)
# ============================================================================





# ============================================================================
# Environment Configs
# ============================================================================

ENV_CONFIGS = {
    "CartPole-v1": {
        "num_generations": 2000,  # 10 tasks x 200 gens
        "pop_size": 512,
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 10,
        "task_interval": 200,
        "num_tasks": 10,
    },
    "Acrobot-v1": {
        "num_generations": 2000,
        "pop_size": 512,
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 10,
        "task_interval": 200,
        "num_tasks": 10,
    },
    "MountainCar-v0": {
        "num_generations": 2000,
        "pop_size": 512,
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 10,
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

def make_scoring_fn(env, policy, param_template, episode_length, num_evals=10,
                    behaviour_cfg=None, traj_steps=10):
    """Create scoring function for continual learning.

    Accepts env_params and noise_vector as arguments (not closure-captured)
    so they can change across tasks.

    Evaluates each individual with num_evals trials:
    - Returns fitness from FIRST trial only (for selection)
    - Returns mean fitness across all trials (for logging/tracking)

    With `behaviour_cfg` set it additionally returns, per individual, the
    sub-sampled observation trajectory of the first trial (what AURORA encodes)
    and the behaviour descriptors averaged over all trials -- the same contract
    as the noncontinual trainer. Selection sees neither, so tracking cannot
    change what the run does.

    The descriptors are built from the *noisy* observations the policy acts on,
    not the clean ones: the sub-task is defined by that offset, and a descriptor
    computed on the clean stream would call two populations identical when they
    are solving different sub-tasks.
    """
    traj_indices = subsample_indices(episode_length, traj_steps)
    num_traj_steps = int(traj_indices.shape[0])
    track = behaviour_cfg is not None

    def evaluate_single(flat_params, eval_key, noise_vector, env_params):
        params = unflatten_params(flat_params, param_template)
        obs, state = env.reset(eval_key, env_params)

        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
            noisy_obs = obs + noise_vector
            logits = policy.apply(params, noisy_obs)
            action = jnp.argmax(logits)  # Deterministic for evaluation

            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)

            total_reward = total_reward + reward * (1.0 - done_flag)
            # Valid = this step belongs to the episode (the env auto-resets on
            # done, so later steps are a fresh episode and must be masked out).
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
        # Sample within the episode that actually happened, so no sampled step
        # is post-termination padding.
        behaviour = rollout_behaviour(all_obs, all_actions, valid, behaviour_cfg)
        return (total_reward,
                all_obs[episode_relative_indices(valid, num_traj_steps)],
                behaviour)

    vmapped_eval = jax.vmap(evaluate_single, in_axes=(0, 0, None, None))

    @jax.jit
    def scoring_fn(flat_genotypes, key, noise_vector, env_params):
        """Returns (fitness_for_selection, mean_fitness_for_logging, extras).

        `extras` is None unless the scoring function was built with a
        BehaviourConfig.
        """
        pop_size = flat_genotypes.shape[0]

        # Always evaluate with num_evals trials per individual
        all_keys = random.split(key, pop_size * num_evals)
        flat_params_repeated = jnp.repeat(flat_genotypes, num_evals, axis=0)
        if track:
            all_fitnesses, all_traj, all_behaviour = vmapped_eval(
                flat_params_repeated, all_keys, noise_vector, env_params)
        else:
            all_fitnesses = vmapped_eval(flat_params_repeated, all_keys,
                                         noise_vector, env_params)
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
# Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='GA on Gymnax (Continual)')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0',
                                 *DEEPSEA_ENV_NAMES])
    parser.add_argument('--num_generations', type=int, default=None)
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--elite_ratio', type=float, default=0.5)
    parser.add_argument('--mutation_std', type=float, default=0.5)
    parser.add_argument('--variation', type=str, default=GAUSSIAN,
                        choices=list(VARIATIONS),
                        help="how offspring are bred. 'gaussian' is this GA as "
                             'published (one drawn archive member plus '
                             "isotropic noise of width --mutation_std). "
                             "'isoline' swaps in DNS's Iso+LineDD two-parent "
                             'operator and changes NOTHING else -- selection is '
                             'still fitness truncation -- so that a DNS-over-GA '
                             'gap can be attributed to novelty selection rather '
                             'than to recombination. Pair it with '
                             '--iso_sigma/--line_sigma and with '
                             '--variation gaussian on the DNS side; see '
                             'source/algorithms/ne/variation.py.')
    parser.add_argument('--iso_sigma', type=float, default=None,
                        help='isoline only; default is the DNS reference value')
    parser.add_argument('--line_sigma', type=float, default=None,
                        help='isoline only; default is the DNS reference value')
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
    parser.add_argument('--noise_range', type=float, default=1.0,
                        help='Scale for observation noise (task_type=noise)')
    parser.add_argument('--noise_type', type=str, default='normal',
                        choices=['normal', 'uniform'],
                        help='Noise distribution: normal (Gaussian) or uniform')
    parser.add_argument('--task_type', type=str, default='noise',
                        choices=['noise', 'param', 'actions'],
                        help='How tasks differ: observation noise or env parameter variation')
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
    parser.add_argument('--no_gifs', action='store_true',
                        help='Skip rendering rollout GIFs. They cost ~20 rollouts plus '
                             'matplotlib per task switch and are not used by any metric; '
                             'scripts/neurips_2026_rebuttal/make_gifs.py renders them '
                             'from the checkpoints afterwards.')
    parser.add_argument('--traj_steps', type=int, default=50,
                        help='Sub-sampled trajectory length fed to AURORA')
    # --track_diversity, --diversity_interval, --occupancy_bins,
    # --behaviour_snapshots and the rest, same flags and defaults as the
    # noncontinual trainer. Here --behaviour_snapshots counts snapshots *per
    # sub-task* rather than over the whole run; see continual_snapshot_gens.
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
    task_interval = args.task_interval
    task_period = args.task_period
    noise_range = args.noise_range
    noise_type = args.noise_type
    task_type = args.task_type
    
    # ONE table for every trainer (source/utils/task_sequence.py). Each of GA,
    # DNS and RL used to carry its own PARAM_CONFIGS literal and they did not
    # agree -- CartPole's range was [0.98, 98.0] here and [0.098, 198.0] in the
    # RL trainer -- so the NE and RL arms of one compute-matched comparison
    # drew different sub-task sequences from the same trial-seeded key.
    param_cfg = GYMNAX_PHYSICS_TASKS.get(env_name, {'param': None, 'mult_range': None})
    param_name = args.param_name or param_cfg['param']
    param_range = args.param_range if args.param_range is not None else param_cfg['mult_range']
    
    output_dir = args.output_dir or f"projects/gymnax/ga_{env_name}_continual_{task_type}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    gifs_dir = os.path.join(output_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)
    mean_gifs_dir = os.path.join(output_dir, "mean_gifs")
    os.makedirs(mean_gifs_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    mean_checkpoints_dir = os.path.join(output_dir, "mean_checkpoints")
    os.makedirs(mean_checkpoints_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, "train.log")
    sys.stdout = Tee(log_file)
    
    print("=" * 60)
    print(f"GA on {env_name} (CONTINUAL)")
    print("=" * 60)
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}")
    print(f"  Task interval: {task_interval} gens")
    print(f"  Task type: {task_type}")
    if task_type == 'noise':
        print(f"  Noise range: {noise_range}")
    else:
        print(f"  Param: {param_name}, range: {param_range}")
    print(f"  Num evals: {num_evals}")
    
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
            if noise_type == 'normal':
                nv = random.normal(noise_key, (obs_dim,)) * noise_range
            else:
                nv = random.uniform(noise_key, (obs_dim,), minval=-noise_range, maxval=noise_range)
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
    
    # Create scoring function (env_params passed as argument, not captured in closure)
    scoring_fn = make_scoring_fn(env, policy, param_template, episode_length, num_evals)
    # The REPORTED evaluation, compiled once for a batch of TWO -- the
    # centroid and the generation's elite, scored together so the pair costs
    # one trace. `report_episodes` rather than `num_evals` on purpose; see
    # --report_episodes.
    report_scoring_fn = make_scoring_fn(env, policy, param_template,
                                        episode_length, report_episodes)

    # A second scoring function that also returns the behaviour descriptors.
    # Kept separate, and called only on the generations the tracker asks for,
    # because collecting them costs an extra pass over every rollout.
    track_diversity = bool(getattr(args, 'track_diversity', False))
    behaviour_cfg = BehaviourConfig(
        env_name=env_name, num_actions=int(action_dim),
        occupancy_bins=args.occupancy_bins) if track_diversity else None
    scoring_fn_bd = make_scoring_fn(
        env, policy, param_template, episode_length, num_evals,
        behaviour_cfg, args.traj_steps) if track_diversity else None
    
    # The variation operator, resolved once so that the run config, the
    # searcher and the arm's name in metrics.yaml cannot disagree.
    variation_params = (
        resolve_params(GAUSSIAN, sigma=args.mutation_std,
                       cross_over_rate=0.0)
        if args.variation == GAUSSIAN else
        resolve_params(ISOLINE, iso_sigma=args.iso_sigma,
                       line_sigma=args.line_sigma))

    # A crossed arm has to be matched to the DNS arm it is being compared
    # against, and this trainer has no ENV_CONFIGS row of iso/line widths to
    # match it from -- the gymnax DNS trainer runs at 0.05 / 0.5 (the earlier
    # study's values), while the operator's own defaults are the DNS paper's
    # corrected 0.005 / 0.05, ten times smaller. Leaving them unset therefore
    # produces a GA/isoline arm at a DIFFERENT operator scale from the `dns`
    # arm, which reintroduces the confound this ablation exists to remove.
    if args.variation == ISOLINE and (args.iso_sigma is None
                                      or args.line_sigma is None):
        print("  WARNING: --variation isoline without explicit "
              "--iso_sigma/--line_sigma. Running at the operator defaults "
              f"({variation_params['iso_sigma']} / "
              f"{variation_params['line_sigma']}). The gymnax DNS arm runs at "
              "0.05 / 0.5; pass those to compare the SELECTION rule rather "
              "than the operator scale.")

    # Initialize wandb
    config = {
        'env': env_name, 'num_generations': num_generations,
        'pop_size': pop_size, 'seed': seed, 'trial': trial,
        'elite_ratio': args.elite_ratio, 'mutation_std': args.mutation_std,
        # Which operator actually bred the offspring. `ga` and `ga_isoline` are
        # one trainer with one string changed, so without this nothing in the
        # run says which of them a directory holds.
        # The RESOLVED widths, not the raw flags: `--iso_sigma` is None
        # unless it was passed, so recording the flag would leave an isoline
        # run claiming a width of null while it bred at the operator's
        # defaults. `resolve_params` also rejects a knob the chosen operator
        # does not have, so this cannot record a scale nothing applied.
        'variation': args.variation, **variation_params,
        # Not a flag any more: the archive is ALWAYS re-scored every
        # generation. Recorded so a run says so outright.
        'refresh_archive': True,
        'num_evals': num_evals, 'report_episodes': report_episodes,
        'hidden_dims': hidden_dims,
        'task_interval': task_interval, 'task_period': task_period,
        'noise_range': noise_range,
        'noise_type': noise_type,
        'task_type': task_type, 'param_name': param_name, 'param_range': param_range,
        'continual': True,
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"ga_{env_name}_continual_{task_type}_pop{pop_size}_trial{trial}", reinit=True)
    
    # Setup GA
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(np.array(devices), axis_names=('p',))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())
    parallel_sharding = NamedSharding(mesh, PartitionSpec('p'))
    
    if pop_size % num_devices != 0:
        old_pop_size = pop_size
        pop_size = (pop_size // num_devices + 1) * num_devices
        print(f"  Warning: Adjusted pop_size from {old_pop_size} to {pop_size}")
    
    # `pop_size` is the EVALUATION BUDGET per generation in both modes, so the
    # arms stay compute-matched with each other and with PPO. What differs is
    # what the budget buys.
    #
    #   default          512 offspring, and the archive keeps the fitness it
    #                    was stored with. `tell` ranks the two against each
    #                    other, which is only meaningful while the sub-task
    #                    that produced the stored number is still the one being
    #                    run.
    #   refresh_archive  256 offspring + the 256-member archive re-scored, all
    #                    on THIS generation's sub-task. Every number `tell`
    #                    compares was measured under the same conditions.
    #
    # The refresh arm therefore buys half the offspring and weaker elitism -- an
    # elite has to re-win against a fresh noisy estimate every generation, and
    # selection here uses a SINGLE episode, so a good genome can be evicted by
    # one unlucky draw. That is the trade the switching study already made
    # deliberately; see GASearcher in source/studies/generalists/ne.py.
    # Unconditional since 2026-09-08. It used to be `--refresh_archive`,
    # defaulting OFF, and with it off an elite kept the fitness it was scored
    # with on a PREVIOUS sub-task -- at the first boundary a stored 500 against
    # an actual 30 -- so `tell` ranked fresh offspring against stale numbers and
    # the archive froze for the rest of the run. There is no question to which
    # that is the answer, so the switch is gone rather than defaulted on. Every
    # `ga` tree on disk dated before then was produced with it off; those runs
    # are superseded by the `ga_refresh` ones and `scripts/make_lineplot.py`
    # drops them.
    #
    # `--reeval_archive`, which re-scored the archive only AT a boundary, is
    # gone with it: it needs to know a switch happened, which CLAUDE.md rule (d)
    # forbids, and re-scoring every generation makes it redundant anyway.
    num_elites = max(1, int(pop_size * args.elite_ratio))
    num_offspring = pop_size - num_elites
    if num_offspring < 1:
        raise SystemExit('a refreshed archive needs --elite_ratio < 1')

    ga = SimpleGA(
        popsize=num_offspring,
        num_dims=num_params,
        # `popsize` is already the offspring count, so the ratio that makes
        # `elite_popsize` come out at `num_elites` is 1.0.
        elite_ratio=1.0,
        sigma_init=args.mutation_std,
        variation=args.variation,
        iso_sigma=args.iso_sigma,
        line_sigma=args.line_sigma,
    )
    ga_params = ga.default_params
    print(f"  Archive refreshed every generation: {num_offspring} "
          f"offspring + {ga.elite_popsize} re-scored elites = "
          f"{num_offspring + ga.elite_popsize} evaluations/generation")

    key, state_init_key = random.split(key)
    ga_state = ga.init(state_init_key, ga_params)

    @jax.jit
    def jit_ask(key, state, params):
        # ask_with_parents, not ask: the plasticity churn column is measured
        # over (parent, offspring) pairs, which is the only pairing that has
        # the same network before and after one update. Identical offspring to
        # ask() for the same key -- both go through SimpleGA._breed -- so this
        # does not change the search. See pairwise_churn in
        # source/metrics/plasticity.py.
        population, new_state, parents = ga.ask_with_parents(key, state, params)
        population = jax.device_put(population, parallel_sharding)
        parents = jax.device_put(parents, parallel_sharding)
        return population, new_state, parents

    @jax.jit
    def jit_tell(population, fitness, state, params):
        return ga.tell(population, fitness, state, params)
    
    # Initialize task 0
    current_task = 0
    noise_vector = jnp.zeros((obs_dim,))
    
    if task_type == 'noise':
        noise_vector = task_noise_vectors[0]
        print(f"\n  Task 0 noise vector ({noise_type}): {jax.device_get(noise_vector)}")
        print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")
    elif task_type == 'param':
        param_val = task_param_values[0]
        env_params = apply_physics(env_name, base_env_params, param_name, param_val)
        print(f"\n  Task 0 {param_name}: {param_val:.4f}x")
    elif task_type == 'actions':
        env_params = FlippedParams(base_env_params, jnp.float32(task_flips[0]))
        print(f"\n  Task 0 action order: "
              f"{'REVERSED' if task_flips[0] else 'stock'}")
    
    # Warmup JIT
    print("\nJIT compiling...")
    key, warmup_key, warmup_ask_key = random.split(key, 3)
    warmup_pop, _, _ = jit_ask(warmup_ask_key, ga_state, ga_params)
    _ = scoring_fn(warmup_pop, warmup_key, noise_vector, env_params)
    print("  JIT compilation complete!")
    
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
    task_gens_to_threshold = None  # gen (relative to task start) when threshold first reached
    all_metrics = []  # list of per-task metric dicts
    training_metrics = []  # Per-generation history, saved to training_metrics.json

    # One agent per sub-task for source/studies/evaluate_continual.py: the best
    # member of the sub-task's final generation, and the GA's elite mean. They
    # are appended together at every task switch, so the two lists stay aligned.
    ckpt_finalgen = []
    ckpt_incumbent = []
    # The network `centroid_fitness` scores: the coordinate-wise mean of the
    # ELITE ARCHIVE. `ckpt_incumbent` is `ga_state.mean`, which this trainer
    # sets to `archive[0]` every generation -- the best archive member, not a
    # mean -- so the two are different networks and the centroid figure needs
    # this one. Both are saved.
    ckpt_centroid = []
    
    # Zero-shot eval for task 0: evaluate random init on task 0
    # Use the initial population's best individual
    key, zt_init_key = random.split(key)
    warmup_pop_zt, _, _ = jit_ask(zt_init_key, ga_state, ga_params)
    warmup_pop_host = jax.device_get(warmup_pop_zt)
    # Just use first individual (all random, no "best" yet)
    zt_params = warmup_pop_host[0]
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
        # snapshots=0 above, then set here: the tracker's own schedule spreads
        # them over the whole run, which would leave whole sub-tasks unsampled.
        tracker.snapshot_gens = continual_snapshot_gens(
            num_generations, task_interval, args.behaviour_snapshots)
        print(f"  Diversity tracking: every {args.diversity_interval} gens, "
              f"{len(tracker.snapshot_gens)} snapshots "
              f"({args.behaviour_snapshots} per sub-task)")

    # Diversity draws from its own stream so that switching tracking on or off
    # leaves the search's random numbers -- and therefore the run -- unchanged.
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

    print(f"\nStarting continual training...")

    for gen in range(num_generations):
        # Check if we need to switch task
        if gen > 0 and gen % task_interval == 0:
            # Use best from the last generation of this task (not best across all task generations)
            eval_params = population_host[int(np.argmax(mean_fitness_host))].copy()
            
            # Per-task evaluation with 10 trials
            print(f"\n  Task {current_task} final evaluation (10 trials)...")
            task_eval_rewards = []
            for eval_trial in range(10):
                key, eval_key_trial = random.split(key)
                _, trial_reward = rollout_for_gif(
                    env, env_params, policy, eval_params, param_template,
                    episode_length, eval_key_trial, noise_vector
                )
                task_eval_rewards.append(trial_reward)
            task_eval_mean = float(np.mean(task_eval_rewards))
            task_eval_std = float(np.std(task_eval_rewards))
            print(f"  Task {current_task} eval: {task_eval_mean:.2f} +/- {task_eval_std:.2f}")
            wandb.summary[f"task_{current_task}_eval_mean"] = task_eval_mean
            wandb.summary[f"task_{current_task}_eval_std"] = task_eval_std
            
            # Create task subfolder and save 10 GIFs
            if not args.no_gifs:
                task_gif_dir = os.path.join(gifs_dir, f"task{current_task}")
                os.makedirs(task_gif_dir, exist_ok=True)
                try:
                    for gif_idx in range(10):
                        key, gif_key = random.split(key)
                        obs_list, total_reward = rollout_for_gif(
                            env, env_params, policy, eval_params, param_template,
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

            # The incumbent carried across the switch. `state.mean` is
            # SimpleGA's best_member, not the archive average, despite the name
            # this block has always used.
            mean_params = jax.device_get(ga_state.mean)
            centroid_params = jax.device_get(ga_state.archive.mean(axis=0))
            
            # Evaluate elite mean (10 trials)
            print(f"  Task {current_task} elite mean evaluation (10 trials)...")
            mean_eval_rewards = []
            for eval_trial in range(10):
                key, eval_key_trial = random.split(key)
                _, trial_reward = rollout_for_gif(
                    env, env_params, policy, mean_params, param_template,
                    episode_length, eval_key_trial, noise_vector
                )
                mean_eval_rewards.append(trial_reward)
            mean_eval_mean = float(np.mean(mean_eval_rewards))
            mean_eval_std = float(np.std(mean_eval_rewards))
            print(f"  Task {current_task} elite mean eval: {mean_eval_mean:.2f} +/- {mean_eval_std:.2f}")

            # The two agents evaluate_continual.py scores for this sub-task.
            ckpt_finalgen.append(np.asarray(eval_params).copy())
            ckpt_incumbent.append(np.asarray(mean_params).copy())
            ckpt_centroid.append(np.asarray(centroid_params).copy())

            # Save mean checkpoint
            mean_ckpt_path = os.path.join(mean_checkpoints_dir, f"task_{current_task}.pkl")
            with open(mean_ckpt_path, 'wb') as f:
                pickle.dump({
                    'flat_params': np.array(mean_params),
                    'eval_mean': mean_eval_mean,
                    'eval_std': mean_eval_std,
                }, f)
            print(f"    Saved mean checkpoint: {mean_ckpt_path}")
            
            # Save 10 GIFs for elite mean
            if not args.no_gifs:
                mean_task_gif_dir = os.path.join(mean_gifs_dir, f"task{current_task}")
                os.makedirs(mean_task_gif_dir, exist_ok=True)
                try:
                    for gif_idx in range(10):
                        key, gif_key = random.split(key)
                        obs_list, total_reward = rollout_for_gif(
                            env, env_params, policy, mean_params, param_template,
                            episode_length, gif_key, noise_vector
                        )
                        frames = []
                        fig, ax = None, None
                        for idx_f, obs in enumerate(obs_list[::2]):
                            step = idx_f * 2
                            frame, fig, ax = render_fn(obs, fig, ax, step=step)
                            frames.append(frame)
                        plt.close(fig)
                        gif_path = os.path.join(mean_task_gif_dir, f"task{current_task}_mean_rollout_{gif_idx:02d}_reward{total_reward:.0f}.gif")
                        imageio.mimsave(gif_path, frames, fps=30, loop=0)
                    print(f"  Saved 10 mean GIFs for task {current_task} in {mean_task_gif_dir}")
                except Exception as e:
                    print(f"  Warning: Failed to save mean GIFs: {e}")

            # Save per-task checkpoint
            task_ckpt_path = os.path.join(checkpoints_dir, f"task_{current_task}.pkl")
            ckpt_data = {
                'flat_params': np.array(eval_params),
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
            if task_type == 'noise':
                ckpt_data['noise_vector'] = jax.device_get(noise_vector)
            elif task_type == 'param':
                ckpt_data['param_name'] = param_name
                ckpt_data['param_mult'] = float(task_param_values[current_task])
            elif task_type == 'actions':
                ckpt_data['action_flip'] = int(task_flips[current_task])
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
                print(f"  Full noise ({noise_type}): {jax.device_get(noise_vector)}")
                print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")
            elif task_type == 'param':
                param_val = task_param_values[current_task]
                env_params = apply_physics(env_name, base_env_params,
                                           param_name, param_val)
                print(f"\n>>> Task {current_task} started at gen {gen}")
                print(f"  {param_name}: {param_val:.4f}x")
            elif task_type == 'actions':
                flip = task_flips[current_task]
                env_params = FlippedParams(base_env_params, jnp.float32(flip))
                print(f"\n>>> Task {current_task} started at gen {gen}")
                print(f"  action order: {'REVERSED' if flip else 'stock'}")
            
            # Zero-shot evaluation on new task (before training)
            zt_rewards = []
            for eval_trial in range(10):
                key, eval_key_trial = random.split(key)
                _, trial_reward = rollout_for_gif(
                    env, env_params, policy, eval_params, param_template,
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
            
            # The archive carries across the boundary, and so do the
            # fitnesses stored with it -- which were measured on the sub-task
            # that just ended. `tell` concatenates this generation's fitnesses
            # with `state.fitness` and keeps the smallest of the union, so if
            # the new sub-task is systematically harder than the old one, no
            # offspring can displace a stale entry and the archive freezes for
            # the rest of the run. Not for a few generations: permanently.
            # `state.best_fitness` is worse again, being a running best over
            # every sub-task so far, and `mean`/`best_member` are set from it.
            # Reset task-specific best tracking for new task
            task_best_fitness = -float('inf')
            task_best_params = None
        
        key, ask_key, eval_key, tell_key = random.split(key, 4)
        
        offspring, ga_state, parents = jit_ask(ask_key, ga_state, ga_params)
        # Under refresh the archive rides along in the evaluated batch, so
        # every genome `tell` ranks was scored on THIS generation's sub-task.
        # Offspring AND the archive, so every number `tell` compares was
        # measured on this generation's sub-task. The batch is `pop_size`, which
        # is what keeps this compute-matched with PPO and keeps every diagnostic
        # below reading the same number of genomes.
        population = jnp.concatenate([offspring, ga_state.archive], axis=0)
        # gen 0 always computes descriptors, so the plasticity probe batch can
        # be frozen from the initial population's visited states.
        measure_now = (tracker is not None and tracker.needs(gen)) or gen == 0
        fitness, mean_fitness, extras = (
            scoring_fn_bd if measure_now else scoring_fn)(
                population, eval_key, noise_vector, env_params)
        
        mean_fitness_host = jax.device_get(mean_fitness)
        population_host = jax.device_get(population)
        
        # SimpleGA MINIMIZES, so negate fitness (use single-trial for selection).
        #
        # `tell` combines the fitnesses it is handed with `state.fitness` and
        # keeps the best `elite_popsize` of the union. Overwriting
        # `state.fitness` with the archive's FRESH scores first is what turns
        # that union into "rank everything measured this generation" -- it is
        # GASearcher.tell under `refresh`, expressed through the class this
        # trainer already uses.
        ga_state = ga_state.replace(fitness=-fitness[num_offspring:])
        ga_state = jit_tell(offspring, -fitness[:num_offspring],
                            ga_state, ga_params)
        # `best_member`/`best_fitness` are a running best over every sub-task
        # so far, and `tell` sets `mean` from them. Left alone they freeze
        # exactly as the archive used to: a score from an easier sub-task is
        # never beaten, so the checkpointed "mean" policy stops moving.
        # GASearcher has no such field -- its incumbent is simply `archive[0]`
        # -- so this restates that here.
        ga_state = ga_state.replace(mean=ga_state.archive[0],
                                    best_member=ga_state.archive[0],
                                    best_fitness=ga_state.fitness[0])
        
        # Track using mean-of-10 fitness
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
        # A GA has no distribution mean, so its centroid is the
        # coordinate-wise mean of the ELITE ARCHIVE -- not a policy the search
        # ever evaluated or would hand back (`archive[0]` is). Read it as
        # "has the archive collapsed onto one solution": where it has, this
        # and `elite_eval_fitness` agree; where they part, the archive is
        # still spread and averaging genuinely different networks gives a
        # network worse than any of them.
        key, report_key = random.split(key)
        report_scores = np.asarray(jax.device_get(report_scoring_fn(
            jnp.stack([ga_state.archive.mean(axis=0), population[best_idx]]),
            report_key, noise_vector, env_params)[1]))
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
        
        log_dict = {
            "generation": gen, "best_fitness": gen_best,
            "mean_fitness": gen_mean, "best_overall": best_overall_fitness,
            "centroid_fitness": centroid_fitness,
            "elite_eval_fitness": elite_eval_fitness,
            "task": current_task,
        }
        if pending_zero_shot is not None:
            attach_zero_shot(log_dict, *pending_zero_shot)
            pending_zero_shot = None
        if task_type == 'noise':
            log_dict["noise_magnitude"] = float(jnp.linalg.norm(noise_vector))
        elif task_type == 'param':
            log_dict[f"{param_name}_mult"] = float(task_param_values[current_task])
        elif task_type == 'actions':
            log_dict['action_flip'] = int(task_flips[current_task])

        # Plasticity: elite churn across generations, within-population churn,
        # and dormancy. See source/metrics/plasticity.py.
        if gen == 0 and measure_now and not plast.started():
            plast.start(plasticity_key, extras["observations"])
        _pop_host = jax.device_get(population)
        _elite_flat = _pop_host[int(np.argmax(np.asarray(mean_fitness_host)))]
        # `offspring` is passed explicitly under refresh for the reason the
        # DNS trainers pass it: `parents[i]` is in correspondence with
        # offspring `i`, and the evaluated batch here is offspring FOLLOWED BY
        # the archive, so pairing the whole batch against `parents` would
        # compare genomes that are not in correspondence at all. Population
        # statistics still read the full batch.
        # `centroid=` measures the same columns on `archive.mean(0)`, the
        # network `centroid_fitness` scores. The elite columns argmax over
        # `concat(offspring, archive)` and so can change lineage between
        # generations; the archive mean cannot, and it is what the centroid
        # lineplot draws.
        log_dict.update(plast.update(
            jax.random.fold_in(plasticity_key, gen), _pop_host, _elite_flat,
            centroid=jax.device_get(ga_state.archive.mean(axis=0)),
            parents=jax.device_get(parents),
            offspring=_pop_host[:num_offspring]))

        # Weight statistics of the searched parameters themselves -- the third
        # plasticity signal CLAUDE.md asks for, alongside dormancy and churn,
        # and the one that catches norms growing without bound across the
        # sub-task sequence. Every generation; see train_GA_gymnax.py.
        log_dict.update(population_weight_stats(_pop_host))

        diversity = None
        if measure_now and tracker is not None:
            if gen == 0:
                tracker.start(probe_key, extras["observations"])
            diversity = tracker.update(
                jax.random.fold_in(diversity_key, gen), gen, population,
                extras["observations"], extras["behaviour"], fitness)
            if diversity:
                log_dict.update(diversity)

        wandb.log(log_dict)
        training_metrics.append({**log_dict, 'elapsed_time': time.time() - start_time})

        if gen % args.log_interval == 0 or gen == num_generations - 1:
            line = (f"Gen {gen:4d} | Task {current_task} | Best: {gen_best:8.2f} "
                    f"| Mean: {gen_mean:8.2f} | Overall: {best_overall_fitness:8.2f}")
            if diversity:
                line += f" | {summarise(diversity)}"
            print(line)
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Best overall: {best_overall_fitness:.2f}")
    print(f"  Total tasks: {current_task + 1}")
    
    # Use best from final generation of this task (not best across all task generations)
    final_best_params = population_host[int(np.argmax(mean_fitness_host))].copy()
    
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

    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"ga_{env_name}_continual_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': np.array(best_params) if best_params is not None else None,
            'param_template': param_template,
            'best_fitness': best_overall_fitness,
            'config': config,
            'final_task': current_task,
        }, f)
    print(f"Saved: {ckpt_path}")
    
    # Final incumbent -- see the note at the per-task block above.
    final_mean_params = jax.device_get(ga_state.mean)
    final_centroid_params = jax.device_get(ga_state.archive.mean(axis=0))
    
    # Evaluate final elite mean (10 trials)
    print(f"\n  Task {current_task} elite mean evaluation (10 trials)...")
    mean_eval_rewards = []
    for eval_trial in range(10):
        key, eval_key_trial = random.split(key)
        _, trial_reward = rollout_for_gif(
            env, env_params, policy, final_mean_params, param_template,
            episode_length, eval_key_trial, noise_vector
        )
        mean_eval_rewards.append(trial_reward)
    mean_eval_mean = float(np.mean(mean_eval_rewards))
    mean_eval_std = float(np.std(mean_eval_rewards))
    print(f"  Task {current_task} elite mean eval: {mean_eval_mean:.2f} +/- {mean_eval_std:.2f}")
    
    # Save final task mean checkpoint
    mean_ckpt_path = os.path.join(mean_checkpoints_dir, f"task_{current_task}.pkl")
    with open(mean_ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': np.array(final_mean_params),
            'eval_mean': mean_eval_mean,
            'eval_std': mean_eval_std,
        }, f)
    print(f"Saved final mean checkpoint: {mean_ckpt_path}")
    
    # Save 10 GIFs for final task elite mean
    if not args.no_gifs:
        mean_task_gif_dir = os.path.join(mean_gifs_dir, f"task{current_task}")
        os.makedirs(mean_task_gif_dir, exist_ok=True)
        try:
            for gif_idx in range(10):
                key, gif_key = random.split(key)
                obs_list, total_reward = rollout_for_gif(
                    env, env_params, policy, final_mean_params, param_template,
                    episode_length, gif_key, noise_vector
                )
                frames = []
                fig, ax = None, None
                for idx_f, obs in enumerate(obs_list[::2]):
                    step = idx_f * 2
                    frame, fig, ax = render_fn(obs, fig, ax, step=step)
                    frames.append(frame)
                plt.close(fig)
                gif_path = os.path.join(mean_task_gif_dir, f"task{current_task}_mean_rollout_{gif_idx:02d}_reward{total_reward:.0f}.gif")
                imageio.mimsave(gif_path, frames, fps=30, loop=0)
            print(f"Saved 10 mean GIFs for final task {current_task} in {mean_task_gif_dir}")
        except Exception as e:
            print(f"Warning: Failed to save final mean GIFs: {e}")

    # The final sub-task's agents, closing the two lists.
    ckpt_finalgen.append(np.asarray(final_best_params).copy())
    ckpt_incumbent.append(np.asarray(final_mean_params).copy())
    ckpt_centroid.append(np.asarray(final_centroid_params).copy())

    # Save final task checkpoint for KL divergence analysis
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
    if task_type == 'noise':
        final_ckpt_data['noise_vector'] = jax.device_get(noise_vector)
    elif task_type == 'param':
        final_ckpt_data['param_name'] = param_name
        final_ckpt_data['param_mult'] = float(task_param_values[current_task])
    elif task_type == 'actions':
        final_ckpt_data['action_flip'] = int(task_flips[current_task])
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
        # The operator is part of the arm's identity: `ga_isoline` is this
        # trainer's fitness truncation over Iso+LineDD offspring, and must not
        # be averaged in with `ga`.
        'method': 'ga' if args.variation == GAUSSIAN else 'ga_isoline',
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
    print(f"Saved {len(training_metrics)} generations to training_metrics.json")

    # Saved populations for the offline cross-method diversity analysis. Written
    # before the GIF rendering below, so a failure there cannot lose them.
    if tracker is not None:
        tracker.save(output_dir)
        print(f"Saved {len(tracker.snapshot_gens)} population snapshots to "
              f"behaviour_snapshots.npz")

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
            method='ga' if args.variation == GAUSSIAN else 'ga_isoline',
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
