"""
Train PPO on Gymnax environments (CONTINUAL).

Task changes every `task_interval` updates by either:
  - Adding observation noise (task_type=noise)
  - Varying an environment parameter (task_type=param):
      CartPole: gravity [0.98, 98.0] (default 9.8, factor of 10)
      MountainCar: gravity [0.000833, 0.0075] (default 0.0025, factor of 3)
      Acrobot: link_length_1 [0.5, 2.0] (default 1.0)

Supports CartPole-v1, Acrobot-v1, MountainCar-v0.
All are discrete action space environments.

Network architecture:
- Policy: 2 hidden layers of 16 neurons each with ReLU
- Value: 3 hidden layers of 128 neurons each with ReLU

Supports:
- PPO: Standard Proximal Policy Optimization
- TRAC: the parameter-free optimizer tuner of Muppidi et al. (NeurIPS 2024),
  applied via `start_trac` from the `trac-optimizer` package -- the same wrapper
  the brax, mujoco and kinetix RL trainers use
- ReDo: Reinitializing Dormant Neurons
- CBP: Continual Backprop (generate-and-test neuron replacement)
- C-CHAIN: Churn Approximated Reduction (Tang et al., 2025), a continual RL
  baseline that regularizes the policy against its own recent past to suppress
  churn; ported from inspiration/C-CHAIN

Usage:
    python train_RL_gymnax_continual.py --env CartPole-v1 --method ppo --gpus 0
    python train_RL_gymnax_continual.py --env Acrobot-v1 --method trac --gpus 0
    python train_RL_gymnax_continual.py --env MountainCar-v0 --method cchain --gpus 0
"""

import argparse
import functools
import os
import sys
import time
import pickle
import json
import yaml

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
from source.algorithms.networks import PolicyNetwork, ValueNetwork
from source.algorithms.rl.ppo import categorical_entropy, categorical_log_prob, categorical_sample, compute_ppo_loss, gae_advantages, make_vec_env_fns, run_redo_pass, train_step, train_step_joint
from source.utils.runtime import Tee, _get_gpu_arg

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
from trac_optimizer.experimental.jax.trac import start_trac
import gymnax
import wandb
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')  # Headless backend for GIF rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.lines import Line2D

from jax.flatten_util import ravel_pytree

from source.algorithms.rl import redo
from source.metrics import plasticity
from source.metrics import ntk as ntk_metrics
from source.metrics.plasticity import policy_churn_cross_entropy as chain_policy_churn
from source.metrics.weight_stats import weight_stats
from source.algorithms.networks import ACTIVATIONS
from source.algorithms.rl.ppo import GYMNAX_POLICY_ACTIVATION, GYMNAX_VALUE_ACTIVATION
from source.metrics.behaviour_descriptors import collect_probe_states
from source.studies.gymnax.cchain import (
    ChainCoefController, add_chain_args, init_chain_state, make_chain_sgd_epochs,
)
from source.envs.gymnax_classic import (
    DEEPSEA_ENV_NAMES, DEEPSEA_SIZES, task_noise_vectors as make_task_noise_vectors,
    make_gymnax_env, wrap_actions)
from source.utils.task_sequence import (
    GYMNAX_PHYSICS_TASKS, action_flip_sequence, cycle_task_sequence,
    physics_mult_sequence)
from source.envs.gymnax_classic import FlipEnv, FlippedParams, apply_physics


# Solved thresholds for Speed-Up (SU) metric
SOLVED_THRESHOLDS = {
    'CartPole-v1': 475,
    'Acrobot-v1': -70,
    'MountainCar-v0': -110,
}
SOLVED_THRESHOLDS.update({f'DeepSea{n}-bsuite': 0.5 for n in DEEPSEA_SIZES})  # the treasure

# ============================================================================
# Logging Helper
# ============================================================================


# ============================================================================
# GIF Rendering Functions
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
    return image


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
    return image


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
    return image


def get_render_fn(env_name):
    """Get the appropriate render function for the environment."""
    if 'CartPole' in env_name:
        return render_cartpole_frame
    elif 'Acrobot' in env_name:
        return render_acrobot_frame
    elif 'MountainCar' in env_name:
        return render_mountaincar_frame
    else:
        return None


# ============================================================================
# Network Definitions
# ============================================================================



# ============================================================================
# PPO Functions
# ============================================================================





# ============================================================================
# Vectorized Environment Step (for parallel rollouts)
# ============================================================================


# ============================================================================
# Rollout and Training
# ============================================================================

def collect_rollout(
    key,
    policy_network,
    policy_params,
    value_network,
    value_params,
    env,
    env_params,
    num_envs,
    num_steps,
    noise_vector,
    env_carry=None,
):
    """Collect rollout data from parallel environments using jax.lax.scan (GPU-friendly).

    `env_carry` is the (obs, states) the previous rollout stopped on, which makes
    a rollout a sliding window over episodes that are still running rather than
    num_steps steps from a fresh reset. See the identical argument in
    train_RL_gymnax.py: resetting here meant every batch was steps 0..49 of a
    new episode, and 0/512 Acrobot or MountainCar episodes terminate that fast,
    so `dones` was all-False and `rewards` constant -- no reward signal at all.

    The caller passes None at each sub-task boundary as well as at the start of
    the run. A boundary changes the observation noise or an env_params physics
    value, i.e. the MDP, so continuing an in-flight episode across one would put
    transitions from two different MDPs in a single GAE trace and hand the first
    rollout of a sub-task data generated under the previous one. Each sub-task's
    training data stays self-contained, which is also what its per-task metrics
    (zero-shot, forward transfer -- all evaluated from reset) assume.

    Args:
        noise_vector: Observation noise for continual learning. Shape: (obs_dim,)
        env_carry: (obs, states) to resume from, or None to reset.

    Returns (rollout, next_carry, last_obs); GAE bootstraps V(last_obs).
    """

    if env_carry is None:
        key, reset_key = random.split(key)
        reset_keys = random.split(reset_key, num_envs)
        obs, states = jax.vmap(lambda k: env.reset(k, env_params))(reset_keys)
    else:
        obs, states = env_carry

    def env_step(carry, _):
        obs, states, key = carry
        
        # Add noise to observations for continual learning
        noisy_obs = obs + noise_vector
        
        # Get policy output (using noisy observations)
        logits = policy_network.apply(policy_params, noisy_obs)
        values = value_network.apply(value_params, noisy_obs)
        
        # Sample actions
        key, action_key = random.split(key)
        action_keys = random.split(action_key, num_envs)
        actions = jax.vmap(categorical_sample)(action_keys, logits)
        log_probs = jax.vmap(categorical_log_prob)(logits, actions)
        
        # Step environment
        key, step_key = random.split(key)
        step_keys = random.split(step_key, num_envs)
        next_obs, next_states, rewards, dones, _ = jax.vmap(
            lambda k, s, a: env.step(k, s, a, env_params)
        )(step_keys, states, actions)
        
        # Handle episode resets - get fresh states for done envs
        key, reset_key = random.split(key)
        reset_keys = random.split(reset_key, num_envs)
        fresh_obs, fresh_states = jax.vmap(lambda k: env.reset(k, env_params))(reset_keys)
        
        # Use fresh state where done
        new_obs = jnp.where(dones[:, None], fresh_obs, next_obs)
        new_states = jax.tree_util.tree_map(
            # broadcast `dones` over every trailing axis of the leaf: DeepSea's
            # state carries a (N, N) action map per env, not only vectors
            lambda fresh, old: jnp.where(dones.reshape((-1,) + (1,) * (fresh.ndim - 1)), fresh, old),
            fresh_states, next_states
        )
        
        # Store transition data (store noisy_obs for training)
        transition = {
            'obs': noisy_obs,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'log_probs': log_probs,
            'values': values,
        }
        
        return (new_obs, new_states, key), transition
    
    # Run scan over num_steps
    (final_obs, final_states, _), rollout = jax.lax.scan(
        env_step, (obs, states, key), None, length=num_steps
    )

    # The carry holds the RAW observation -- the noise is a view applied on the
    # way into the networks, not part of the env state, and a sub-task boundary
    # changes it. The bootstrap observation, though, must be noisy: the value
    # network was applied to `noisy_obs` at every step of the rollout above, so
    # feeding it a clean observation here would evaluate V at a point off the
    # distribution it just fitted.
    last_obs = final_obs + noise_vector

    # rollout is a dict with arrays of shape (num_steps, num_envs, ...)
    return rollout, (final_obs, final_states), last_obs




# ============================================================================
# ReDo lives in source/algorithms/rl/redo.py, shared with train_RL_gymnax.py.
# ============================================================================


# ============================================================================
# Continual Backprop: Generate-and-Test neuron replacement
# ============================================================================

def get_policy_activations(policy_params, obs, hidden_dims):
    """Manual forward pass through policy network to extract hidden activations."""
    x = obs
    activations = []
    p = policy_params['params']
    for i in range(len(hidden_dims)):
        layer_key = f'Dense_{i}'
        x = x @ p[layer_key]['kernel'] + p[layer_key]['bias']
        x = jax.nn.relu(x)
        activations.append(x)
    return activations


def count_dormant_neurons(policy_params, obs, hidden_dims, tau=redo.DEFAULT_TAU):
    """Count dormant neurons per hidden layer, using ReDo's own criterion.

    A neuron is dormant if its mean absolute activation, divided by the layer
    mean, is <= tau. This is the score from Sokar et al. (2023); it replaced an
    earlier max-normalised variant here so that the plasticity diagnostic logged
    for every method counts exactly the neurons `--method redo` recycles.

    Returns:
        per_layer: list of (num_dormant, layer_size) per hidden layer
        total_dormant: total dormant neurons across all layers
        total_neurons: total hidden neurons across all layers
    """
    stats = redo.dormant_stats(policy_params, obs, len(hidden_dims), tau)
    return stats['per_layer'], stats['dormant_count'], stats['total_neurons']


class CBPTracker:
    """Tracks neuron utility for Continual Backprop's generate-and-test.

    Implements the algorithm from Dohare et al. (2024) 'Loss of Plasticity in
    Deep Continual Learning'. Each hidden neuron has a utility score maintained
    as an exponential moving average of its contribution (outgoing weight
    magnitude * activation magnitude). Neurons with the lowest utility that
    have exceeded a maturity threshold are periodically replaced.
    """

    def __init__(self, hidden_dims, decay_rate=0.99):
        self.hidden_dims = hidden_dims
        self.decay_rate = decay_rate
        self.num_layers = len(hidden_dims)
        self.utility = [jnp.zeros(d) for d in hidden_dims]
        self.ages = [jnp.zeros(d) for d in hidden_dims]
        self.mean_act = [jnp.zeros(d) for d in hidden_dims]

    def update_utility(self, layer_idx, activations, policy_params):
        """Update contribution-based utility for one hidden layer."""
        self.ages[layer_idx] = self.ages[layer_idx] + 1
        self.utility[layer_idx] = self.utility[layer_idx] * self.decay_rate
        self.mean_act[layer_idx] = self.mean_act[layer_idx] * self.decay_rate

        # Update mean activation (EMA)
        self.mean_act[layer_idx] = (
            self.mean_act[layer_idx]
            + (1 - self.decay_rate) * activations.mean(axis=0)
        )

        # Contribution utility: outgoing_weight_magnitude * activation_magnitude
        p = policy_params['params']
        next_layer_key = f'Dense_{layer_idx + 1}'
        # Flax kernel shape: (in_features, out_features)
        # For neuron j, outgoing weights = kernel[j, :]
        output_weight_mag = jnp.abs(p[next_layer_key]['kernel']).mean(axis=1)
        new_util = output_weight_mag * jnp.abs(activations).mean(axis=0)

        self.utility[layer_idx] = (
            self.utility[layer_idx] + (1 - self.decay_rate) * new_util
        )

    def get_bias_corrected_utility(self, layer_idx):
        bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]
        return self.utility[layer_idx] / jnp.maximum(bias_correction, 1e-8)


def apply_cbp(policy_state, cbp_tracker, obs_batch, key,
              replacement_rate, maturity_threshold):
    """Apply Continual Backprop: update utility and replace low-utility neurons.

    Returns updated (policy_state, cbp_tracker, total_neurons_replaced).
    """
    policy_params = policy_state.params
    hidden_dims = cbp_tracker.hidden_dims

    # Forward pass to get hidden activations
    activations = get_policy_activations(policy_params, obs_batch, hidden_dims)

    # Update utility for all hidden layers
    for i in range(cbp_tracker.num_layers):
        cbp_tracker.update_utility(i, activations[i], policy_params)

    # Deep-copy the mutable param dicts
    new_params = {}
    for lk, lv in policy_params['params'].items():
        new_params[lk] = {pk: pv for pk, pv in lv.items()}

    total_replaced = 0
    for i in range(cbp_tracker.num_layers):
        ages = cbp_tracker.ages[i]
        eligible_mask = ages > maturity_threshold
        num_eligible = int(jnp.sum(eligible_mask))
        if num_eligible == 0:
            continue

        num_to_replace = replacement_rate * num_eligible

        # Stochastic rounding for fractional counts
        key, subkey = random.split(key)
        if num_to_replace < 1:
            num_to_replace_int = 1 if float(random.uniform(subkey)) < num_to_replace else 0
        else:
            num_to_replace_int = int(num_to_replace)
        if num_to_replace_int == 0:
            continue

        # Select lowest-utility neurons among eligible ones
        bc_utility = cbp_tracker.get_bias_corrected_utility(i)
        utility_for_selection = jnp.where(eligible_mask, bc_utility, jnp.inf)
        replace_indices = jnp.argsort(utility_for_selection)[:num_to_replace_int]

        # --- Reset incoming weights (current layer) ---
        layer_key_name = f'Dense_{i}'
        kernel = new_params[layer_key_name]['kernel']  # (in_features, hidden_dim)
        bias = new_params[layer_key_name]['bias']      # (hidden_dim,)

        in_features = kernel.shape[0]
        # Kaiming uniform for ReLU: bound = sqrt(2) * sqrt(3 / fan_in)
        bound = float(jnp.sqrt(6.0 / in_features))

        key, init_key = random.split(key)
        new_weights = random.uniform(
            init_key, (in_features, num_to_replace_int),
            minval=-bound, maxval=bound,
        )
        kernel = kernel.at[:, replace_indices].set(new_weights)
        bias = bias.at[replace_indices].set(0.0)
        new_params[layer_key_name]['kernel'] = kernel
        new_params[layer_key_name]['bias'] = bias

        # --- Correct next-layer bias and reset outgoing weights ---
        next_layer_key_name = f'Dense_{i + 1}'
        next_kernel = new_params[next_layer_key_name]['kernel']  # (hidden_dim, next_dim)
        next_bias = new_params[next_layer_key_name]['bias']      # (next_dim,)

        # Bias correction: compensate for zeroed-out outgoing weights
        bias_correction = jnp.maximum(
            1 - cbp_tracker.decay_rate ** cbp_tracker.ages[i][replace_indices], 1e-8
        )
        corrected_mean_act = (
            cbp_tracker.mean_act[i][replace_indices] / bias_correction
        )  # (num_replace,)
        # next_kernel[replace_indices, :] shape: (num_replace, next_dim)
        # correction per output neuron: sum over replaced neurons
        bias_delta = (next_kernel[replace_indices, :] * corrected_mean_act[:, None]).sum(axis=0)
        next_bias = next_bias + bias_delta

        next_kernel = next_kernel.at[replace_indices, :].set(0.0)
        new_params[next_layer_key_name]['kernel'] = next_kernel
        new_params[next_layer_key_name]['bias'] = next_bias

        # Reset tracker state for replaced neurons
        cbp_tracker.utility[i] = cbp_tracker.utility[i].at[replace_indices].set(0.0)
        cbp_tracker.ages[i] = cbp_tracker.ages[i].at[replace_indices].set(0.0)
        cbp_tracker.mean_act[i] = cbp_tracker.mean_act[i].at[replace_indices].set(0.0)

        total_replaced += num_to_replace_int

    new_policy_params = {'params': new_params}
    policy_state = policy_state.replace(params=new_policy_params)
    return policy_state, cbp_tracker, total_replaced


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(key, policy_network, policy_params, env, env_params, noise_vector, num_episodes=10, max_steps=500):
    """Evaluate policy on environment with noise (vectorized with jax.lax.scan)."""
    
    # Reset all eval episodes in parallel
    reset_keys = random.split(key, num_episodes + 1)
    key = reset_keys[0]
    episode_keys = reset_keys[1:]
    obs_all, state_all = jax.vmap(lambda k: env.reset(k, env_params))(episode_keys)
    
    def eval_step(carry, _):
        obs, state, key, rewards, dones_acc = carry
        noisy_obs = obs + noise_vector
        logits = policy_network.apply(policy_params, noisy_obs)
        actions = jnp.argmax(logits, axis=-1)
        
        key, step_key = random.split(key)
        step_keys = random.split(step_key, num_episodes)
        next_obs, next_state, reward, done, _ = jax.vmap(
            lambda k, s, a: env.step(k, s, a, env_params)
        )(step_keys, state, actions)
        
        # Only accumulate reward if episode hasn't already ended
        still_running = 1.0 - dones_acc
        rewards = rewards + reward * still_running
        dones_acc = jnp.maximum(dones_acc, done.astype(jnp.float32))
        
        return (next_obs, next_state, key, rewards, dones_acc), None
    
    init_rewards = jnp.zeros(num_episodes)
    init_dones = jnp.zeros(num_episodes)
    
    (_, _, _, total_rewards, _), _ = jax.lax.scan(
        eval_step, (obs_all, state_all, key, init_rewards, init_dones), None, length=max_steps
    )
    
    return jnp.mean(total_rewards), jnp.std(total_rewards)


# ============================================================================
# Environment-Specific Hyperparameters
# ============================================================================

ENV_CONFIGS = {
    # Steps budget matched to GA: pop_size(512) * episode_length(500) * num_gens(200) * num_tasks(10) = 512M
    # steps_per_update = num_envs(2048) * num_steps(50) = 102,400
    # task_interval = steps_per_task(51.2M) / steps_per_update(102.4K) = 500
    "CartPole-v1": {
        "num_timesteps": 512 * 200 * 500 * 10,  # 512M total env steps (matched to GA)
        "task_interval": 500,  # 51.2M / (2048*50) = 500 updates per task
        "num_envs": 2048,
        "num_steps": 50,  # unroll_length
        "num_epochs": 10,  # num_updates_per_batch (SGD passes, doesn't change env steps)
        "num_minibatches": 32,
        "gamma": 0.95,  # discounting
        "learning_rate": 3e-4,
        "ent_coef": 1e-2,
        "policy_hidden_dims": (16, 16),
        "value_hidden_dims": (128, 128, 128),
        "episode_length": 500,
    },
    "Acrobot-v1": {
        "num_timesteps": 512 * 200 * 500 * 10,  # 512M total env steps (matched to GA)
        "task_interval": 500,  # 51.2M / (2048*50) = 500 updates per task
        "num_envs": 2048,
        "num_steps": 50,  # unroll_length
        "num_epochs": 10,  # num_updates_per_batch (SGD passes, doesn't change env steps)
        "num_minibatches": 32,
        "gamma": 0.99,  # discounting
        "learning_rate": 1e-4,
        "ent_coef": 1e-2,
        "policy_hidden_dims": (16, 16),
        "value_hidden_dims": (128, 128, 128),
        "episode_length": 500,
    },
    "MountainCar-v0": {
        "num_timesteps": 512 * 200 * 500 * 10,  # 512M total env steps (matched to GA)
        "task_interval": 500,  # 51.2M / (2048*50) = 500 updates per task
        "num_envs": 2048,
        "num_steps": 50,  # unroll_length
        "num_epochs": 10,  # num_updates_per_batch (SGD passes, doesn't change env steps)
        "num_minibatches": 32,
        # 0.99, matching the stationary trainer, which measured it on
        # 2026-08-06: at the full budget over 3 seeds, 0.95 gave -114.4 +- 11.8
        # with a 20.4-point slide from its peak against 0.99's -97.2 and 8.1.
        # These two files must agree on this: forward transfer is the continual
        # run minus the stationary reference, so a discount that differs between
        # them lands in the FT column as if it were a continual-learning effect.
        "gamma": 0.99,  # discounting
        "learning_rate": 3e-4,
        "ent_coef": 1e-2,
        "policy_hidden_dims": (16, 16),
        "value_hidden_dims": (128, 128, 128),
        "episode_length": 500,
    },
}
# DeepSea<N> (source/envs/gymnax_classic.py DeepSeaEnv): CartPole's settings at
# gamma 0.99; the episode is N steps. The launcher passes the compute-matched
# budget and task interval explicitly (NE_EPISODE_LENGTH=N there), so the
# `num_timesteps` / `task_interval` here are stand-alone defaults only.
for _n in DEEPSEA_SIZES:
    ENV_CONFIGS[f"DeepSea{_n}-bsuite"] = {
        "num_timesteps": 512 * 3 * _n * 4000,
        "task_interval": max(1, 512 * 3 * _n * 200 // (2048 * 50)),
        "num_envs": 2048,
        "num_steps": 50,
        "num_epochs": 10,
        "num_minibatches": 32,
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "ent_coef": 1e-2,
        "policy_hidden_dims": (16, 16),
        "value_hidden_dims": (128, 128, 128),
        "episode_length": _n,
    }


# ============================================================================
# Main Training Loop
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='PPO on Gymnax (Continual Learning)')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0',
                                 *DEEPSEA_ENV_NAMES])
    parser.add_argument('--method', type=str, default='ppo',
                        choices=['ppo', 'trac', 'redo', 'cbp', 'cchain'])
    parser.add_argument('--num_timesteps', type=int, default=None,
                        help='Override default timesteps for env')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)
    
    # Continual learning parameters
    parser.add_argument('--task_interval', type=int, default=None,
                        help='Change task every N updates (default: env-specific for 10 tasks)')
    parser.add_argument('--task_period', type=int, default=0,
                        help='Revisit sub-tasks: after this many distinct ones '
                             'the sequence cycles, so a 20-sub-task run at '
                             'period 10 sees each sub-task twice. 0 (default) '
                             'means every sub-task is new, which is every run '
                             'made before this flag existed.')
    parser.add_argument('--cchain_reset_on_switch', type=int, default=0,
                        help='Re-calibrate the C-CHAIN coefficient controller at '
                             'each sub-task boundary. OFF by default, and the '
                             'default is the point: every other method here is '
                             'never told a switch happened, and this would hand '
                             'C-CHAIN alone that signal. The controller sets its '
                             'coefficient from the running ratio of policy loss to '
                             'churn loss, so re-calibrating is the better estimate '
                             'once the task changes -- which is why it is kept as '
                             'an option. Matches --cchain_reset_on_switch in '
                             'source/studies/brax/my_brax/cchain.py.')
    parser.add_argument('--reset_on_switch', action='store_true',
                        help='Re-initialise the agent and its optimiser at '
                             'every sub-task boundary. The plasticity upper '
                             'bound -- a learner that never carries old '
                             'weights, so it cannot forget and cannot '
                             'transfer. C-CHAIN reports this as its oracle.')
    parser.add_argument('--noise_range', type=float, default=1.0,
                        help='Scale for observation noise (task definition)')
    parser.add_argument('--task_type', type=str, default='noise',
                        choices=['noise', 'param', 'actions'],
                        help='Type of task variation: noise (obs offset), param '
                             '(rescale a physics group) or actions (REVERSE the '
                             'action order on alternate sub-tasks, observation and '
                             'body untouched). `actions` is the only one of the '
                             'three with no generalist for a memoryless policy: '
                             'the two regimes want opposite outputs at the same '
                             'input, so a switch cannot be absorbed by '
                             'interpolating, only relearned.')
    parser.add_argument('--param_name', type=str, default=None,
                        help='Which physics group a param sub-task rescales -- a key of '
                             'PHYSICS_PARAMS[env] in source/envs/gymnax_classic.py. '
                             'Default: the env entry in GYMNAX_PHYSICS_TASKS.')
    parser.add_argument('--param_range', type=float, nargs=2, default=None,
                        help='MULTIPLIER range [min, max] for a param sub-task, drawn '
                             'log-uniformly. NOT an absolute parameter range: sub-task 0 '
                             'is always 1.0x, the stock body. Default: env-specific.')
    
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
    
    # TRAC (Muppidi et al., NeurIPS 2024) takes no hyperparameters -- that is the
    # point of it. `--method trac` wraps the optimizer with `start_trac` and the
    # tuner's own constants (eps, s_prev, num_betas) are left at the package
    # defaults, which is also what brax, mujoco and kinetix pass.

    # ReDo specific (Sokar et al. 2023; see source/algorithms/rl/redo.py). --redo_tau
    # also sets the threshold of the dormant-neuron diagnostic logged for every
    # method, so it is meaningful outside --method redo too.
    parser.add_argument('--redo_interval', type=int, default=50,
                        help='Apply ReDo every N updates')
    parser.add_argument('--num_probe_states', type=int, default=512,
                        help='Size of the frozen probe batch for the churn and dormancy '
                             'diagnostics; same name/default as the NE trainers.')
    parser.add_argument('--redo_tau', type=float, default=redo.DEFAULT_TAU,
                        help='Dormancy threshold on the layer-normalised activation '
                             'score (reference default 0.025; 0 = strictly dead only)')
    parser.add_argument('--redo_targets', type=str, default='both',
                        choices=['both', 'policy', 'value'],
                        help='Which networks to recycle')
    parser.add_argument('--redo_batch_size', type=int, default=512,
                        help='Observations sampled from the rollout to score dormancy')
    parser.add_argument('--redo_keep_adam_count', action='store_true',
                        help='Do not reset the Adam step count (the reference resets it, '
                             'and reports the reset as important for performance)')

    # CBP (Continual Backprop) specific
    parser.add_argument('--cbp_replacement_rate', type=float, default=0.001,
                        help='Fraction of eligible neurons to replace per step')
    parser.add_argument('--cbp_decay_rate', type=float, default=0.99,
                        help='Decay rate for utility EMA')
    parser.add_argument('--cbp_maturity_threshold', type=int, default=100,
                        help='Neuron age before eligible for replacement')

    # C-CHAIN specific
    add_chain_args(parser)

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='continual_neuroevolution_gymnax')
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--no_gifs', action='store_true',
                        help='Skip GIF rendering (much faster for diagnostic runs)')
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    
    return parser.parse_args()


def get_hyperparams(args):
    """Get hyperparameters, using env-specific defaults when not specified."""
    env_config = ENV_CONFIGS[args.env]
    
    return {
        'num_timesteps': args.num_timesteps if args.num_timesteps is not None else env_config['num_timesteps'],
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
    method = args.method
    seed = args.seed + args.trial  # Different seed per trial
    trial = args.trial
    noise_range = args.noise_range
    task_type = args.task_type
    
    # ONE table for every trainer (source/utils/task_sequence.py). This
    # trainer's PARAM_CONFIGS literal did not match the GA and DNS trainers'
    # -- CartPole [0.098, 198.0] here against [0.98, 98.0] there, MountainCar
    # [0.00125, 0.005] against [0.000833, 0.0075] -- so the RL and NE arms of
    # one compute-matched comparison drew DIFFERENT sub-task sequences from
    # the same trial-seeded key, which is exactly what that seeding exists to
    # prevent (CLAUDE.md rule c).
    param_cfg = GYMNAX_PHYSICS_TASKS.get(env_name, {'param': None, 'mult_range': None})
    param_name = args.param_name or param_cfg['param']
    param_range = args.param_range if args.param_range is not None else param_cfg['mult_range']
    
    # Get environment-specific hyperparameters
    hp = get_hyperparams(args)
    num_timesteps = hp['num_timesteps']
    task_interval = hp['task_interval']
    task_period = args.task_period
    reset_on_switch = args.reset_on_switch
    cchain_reset_on_switch = bool(args.cchain_reset_on_switch)
    
    print("=" * 60)
    print(f"PPO ({method.upper()}) on {env_name} (CONTINUAL)")
    print("=" * 60)
    
    # Create environment
    env, env_params = make_gymnax_env(env_name)
    env_params = env_params.replace(max_steps_in_episode=hp['episode_length'])
    # The stock body, kept aside. A param sub-task is a multiplier applied to
    # THIS, not to whatever the previous sub-task left behind.
    base_env_params = env_params
    obs_dim = env.observation_space(env_params).shape[0]
    action_dim = env.action_space(env_params).n
    # AFTER the spaces are read, so they are read off the raw gymnax env with
    # bare params. FlipEnv reverses the action inside `step` under a
    # FlippedParams whose `flip` is a traced scalar, so a switching run still
    # compiles once; `__getattr__` forwards everything else to the wrapped env.
    if task_type == 'actions':
        env = wrap_actions(env)
    
    print(f"  Environment: {env_name}")
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"  Method: {method.upper()}")
    print(f"  Total timesteps: {num_timesteps:,}")
    print(f"  Task interval: {task_interval} updates")
    if task_type == 'noise':
        print(f"  Task type: noise, range: {noise_range}")
    elif task_type == 'param':
        print(f"  Task type: param ({param_name}), range: {param_range}")
    elif task_type == 'actions':
        print(f"  Task type: actions (action order reversed on alternate sub-tasks)")
    print(f"  Hyperparams: num_envs={hp['num_envs']}, num_steps={hp['num_steps']}, "
          f"lr={hp['learning_rate']}, gamma={hp['gamma']}, ent_coef={hp['ent_coef']}")
    
    # Output directory
    if args.output_dir is None:
        output_dir = os.path.join(
            REPO_ROOT, "projects", "gymnax",
            f"{method}_{env_name.replace('-', '_')}_continual_{task_type}",
            f"trial_{trial}"
        )
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, "train.log")
    sys.stdout = Tee(log_file)
    
    print(f"  Output: {output_dir}")
    
    # Create gifs and checkpoints directories
    gifs_dir = os.path.join(output_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Initialize random key
    key = random.key(seed + trial * 1000)
    
    # Create networks using env-specific hidden dims
    policy_network = PolicyNetwork(hidden_dims=hp['policy_hidden_dims'], action_dim=action_dim)
    value_network = ValueNetwork(hidden_dims=hp['value_hidden_dims'])
    
    key, policy_key, value_key = random.split(key, 3)
    dummy_obs = jnp.zeros((1, obs_dim))
    policy_params = policy_network.init(policy_key, dummy_obs)
    value_params = value_network.init(value_key, dummy_obs)
    
    # Count parameters
    policy_param_count = sum(p.size for p in jax.tree_util.tree_leaves(policy_params))
    value_param_count = sum(p.size for p in jax.tree_util.tree_leaves(value_params))
    print(f"  Policy params: {policy_param_count:,}, Value params: {value_param_count:,}")
    
    # Create optimizers.
    #
    # TRAC (Muppidi et al., NeurIPS 2024) is an optimiser wrapper, not a loss or
    # an exploration term, so it goes here and nowhere else. Its tuner state --
    # in particular `theta_ref`, the reference point every subsequent update is
    # measured against -- is created once for the whole run and is deliberately
    # NOT reset at a sub-task boundary: adapting across a non-stationarity it is
    # never told about is the property being measured. Same wiring as
    # `my_brax/ppo_continual_train.py` and `kinetix/experiments/ppo.py`.
    #
    # ONE tuner over the policy AND the value parameters -- see the long note in
    # `train_RL_gymnax.py`. Those two suites are joint by construction (one
    # actor-critic network, one optimiser) and so is the C-CHAIN classic-control
    # reference; a policy-only tuner resets the policy to its initial weights
    # mid-run, which is what the -500 cliffs in the stationary runs were.
    def make_base_optimizer():
        return optax.adam(hp['learning_rate'])

    joint_trac_tx = None
    if method == 'trac':
        joint_trac_tx = start_trac(make_base_optimizer())
        print("  Optimizer: adam wrapped in TRAC (joint policy+value)")

    policy_optimizer = make_base_optimizer()
    value_optimizer = make_base_optimizer()

    policy_state = TrainState.create(
        apply_fn=policy_network.apply,
        params=policy_params,
        tx=policy_optimizer,
    )
    value_state = TrainState.create(
        apply_fn=value_network.apply,
        params=value_params,
        tx=value_optimizer,
    )
    joint_opt_state = None
    if joint_trac_tx is not None:
        joint_opt_state = joint_trac_tx.init(
            {'policy': policy_state.params, 'value': value_state.params})

    # Initialize wandb
    config = {
        'env': env_name, 'method': method, 'num_timesteps': num_timesteps,
        'seed': seed, 'trial': trial, 'num_envs': hp['num_envs'],
        'num_steps': hp['num_steps'], 'learning_rate': hp['learning_rate'],
        'gamma': hp['gamma'], 'ent_coef': hp['ent_coef'],
        'num_epochs': hp['num_epochs'], 'num_minibatches': hp['num_minibatches'],
        'policy_hidden_dims': hp['policy_hidden_dims'], 'value_hidden_dims': hp['value_hidden_dims'],
        'task_interval': task_interval, 'task_period': task_period,
        'reset_on_switch': reset_on_switch,
        'cchain_reset_on_switch': cchain_reset_on_switch,
        'task_type': task_type,
        'noise_range': noise_range, 'param_name': param_name, 'param_range': param_range,
    }
    if method == 'cchain':
        config.update({
            'chain_target_rel_scale': args.chain_target_rel_scale,
            'chain_warmup_updates': args.chain_warmup_updates,
            'chain_coef_window': args.chain_coef_window,
        })
    config['dormant_tau'] = args.redo_tau
    if method == 'redo':
        config.update({
            'redo_interval': args.redo_interval,
            'redo_targets': args.redo_targets,
            'redo_batch_size': args.redo_batch_size,
            'redo_reset_adam_count': not args.redo_keep_adam_count,
        })
        print(f"  ReDo: every {args.redo_interval} updates, tau={args.redo_tau}, "
              f"targets={args.redo_targets}, "
              f"reset_adam_count={not args.redo_keep_adam_count}")
    wandb.init(project=args.wandb_project, config=config,
               name=f"{method}_{env_name}_continual_{task_type}_epochs{hp['num_epochs']}_trial{trial}", reinit=True)
    
    # `ent_coef` is a plain constant for every method here. It used to be state
    # that `--method trac` mutated; see the note on start_trac above.
    ent_coef = hp['ent_coef']

    # C-CHAIN: Initialize reference networks and coefficient controller
    chain_state = None
    chain_ctrl = None
    if method == 'cchain':
        chain_state = init_chain_state(policy_state, value_state)
        chain_ctrl = ChainCoefController(
            args.chain_target_rel_scale, args.chain_warmup_updates, args.chain_coef_window
        )
        print(f"  C-CHAIN: target_rel_scale={args.chain_target_rel_scale}, "
              f"warmup_updates={args.chain_warmup_updates}, "
              f"coef_window={args.chain_coef_window}")

    # CBP: Initialize tracker
    cbp_tracker = None
    if method == 'cbp':
        cbp_tracker = CBPTracker(hp['policy_hidden_dims'], args.cbp_decay_rate)
        print(f"  CBP: replacement_rate={args.cbp_replacement_rate}, "
              f"decay_rate={args.cbp_decay_rate}, "
              f"maturity_threshold={args.cbp_maturity_threshold}")
    
    # Training metrics
    best_reward = -float('inf')
    task_best_reward = -float('inf')
    training_metrics = []
    start_time = time.time()
    
    # Calculate number of updates
    timesteps_per_update = hp['num_envs'] * hp['num_steps']
    num_updates = num_timesteps // timesteps_per_update

    # Run whole sub-tasks only. Any remainder past the last complete sub-task
    # would switch into a sub-task that was never generated, and it would also
    # give the final sub-task a different length from the others.
    leftover = num_updates % task_interval
    if leftover:
        print(f"  Dropping {leftover} update(s): the budget is not a whole "
              f"number of {task_interval}-update sub-tasks")
        num_updates -= leftover

    if num_updates < task_interval:
        raise SystemExit(
            f"Budget of {num_timesteps:,} steps is {num_updates + leftover} update(s), "
            f"less than one {task_interval}-update sub-task. Raise --num_timesteps "
            f"or lower --task_interval."
        )

    print(f"\nStarting continual training ({num_updates} updates)...")
    
    # JIT compile rollout collection (includes noise_vector and env_params)
    @jax.jit
    def jit_collect_rollout_fn(key, policy_params, value_params, noise_vector,
                               env_params, env_carry):
        return collect_rollout(
            key, policy_network, policy_params,
            value_network, value_params,
            env, env_params, hp['num_envs'], hp['num_steps'],
            noise_vector, env_carry=env_carry,
        )

    # Used at the start of the run and again at every sub-task boundary.
    # env_params is an argument because a boundary can change a physics value.
    @jax.jit
    def jit_init_env_carry(key, env_params):
        reset_keys = random.split(key, hp['num_envs'])
        return jax.vmap(lambda k: env.reset(k, env_params))(reset_keys)


    # JIT compile SGD epochs (replaces Python for-loops over epochs/minibatches)
    num_minibatches = hp['num_minibatches']
    num_epochs = hp['num_epochs']
    batch_size = hp['num_steps'] * hp['num_envs']
    minibatch_size = batch_size // num_minibatches
    
    @jax.jit
    def jit_sgd_epochs(policy_state, value_state, opt_state, flat_batch, ent_coef, key):
        """Run num_epochs of SGD with num_minibatches each, fully compiled.

        `opt_state` is the joint TRAC state and is carried through the scan
        alongside the two TrainStates; it is None (and inert) for every other
        method, which keeps using each TrainState's own optimiser.
        """

        def single_epoch(carry, _):
            policy_state, value_state, opt_state, key = carry
            key, shuffle_key = random.split(key)
            perm = random.permutation(shuffle_key, batch_size)
            shuffled = {k: v[perm] for k, v in flat_batch.items()}

            # Reshape into (num_minibatches, minibatch_size, ...)
            mb_data = {
                k: v.reshape(num_minibatches, minibatch_size, *v.shape[1:])
                for k, v in shuffled.items()
            }

            def single_minibatch(carry, minibatch):
                policy_state, value_state, opt_state = carry
                if joint_trac_tx is not None:
                    policy_state, value_state, opt_state, loss, metrics = train_step_joint(
                        policy_network, value_network,
                        hp['clip_eps'], hp['vf_coef'], joint_trac_tx,
                        policy_state, value_state, opt_state, minibatch, ent_coef,
                    )
                else:
                    policy_state, value_state, loss, metrics = train_step(
                        policy_network, value_network,
                        hp['clip_eps'], hp['vf_coef'],
                        policy_state, value_state, minibatch, ent_coef,
                    )
                return (policy_state, value_state, opt_state), metrics

            (policy_state, value_state, opt_state), all_metrics = jax.lax.scan(
                single_minibatch, (policy_state, value_state, opt_state), mb_data,
                length=num_minibatches
            )

            return (policy_state, value_state, opt_state, key), all_metrics

        (policy_state, value_state, opt_state, key), epoch_metrics = jax.lax.scan(
            single_epoch, (policy_state, value_state, opt_state, key), None,
            length=num_epochs
        )

        # Return last metrics (last epoch, last minibatch)
        last_metrics = jax.tree_util.tree_map(lambda x: x[-1, -1], epoch_metrics)
        last_loss = last_metrics.get('pg_loss', 0.0) + hp['vf_coef'] * last_metrics.get('vf_loss', 0.0)

        return policy_state, value_state, opt_state, last_loss, last_metrics

    jit_sgd_epochs_chain = None
    if method == 'cchain':
        jit_sgd_epochs_chain = make_chain_sgd_epochs(
            policy_network, value_network, compute_ppo_loss,
            hp['clip_eps'], hp['vf_coef'], num_epochs, num_minibatches, batch_size,
        )

    # JIT compile vectorized GAE computation
    @jax.jit
    def compute_advantages_returns(rewards, values, dones, last_values):
        """Vectorized GAE computation over all environments."""
        # Input shapes: (num_steps, num_envs); last_values is (num_envs,)
        rewards_T = rewards.T  # (num_envs, num_steps)
        values_T = values.T
        dones_T = dones.T

        advantages_T = jax.vmap(
            lambda r, v, d, lv: gae_advantages(
                r, v, d, hp['gamma'], hp['gae_lambda'], last_value=lv
            )
        )(rewards_T, values_T, dones_T, last_values)
        returns_T = advantages_T + values_T
        
        # Transpose back and normalize
        advantages = advantages_T.T
        returns = returns_T.T
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
    
    # JIT compile evaluation
    @jax.jit
    def jit_evaluate(key, policy_params, noise_vector, env_params):
        return evaluate(
            key, policy_network, policy_params,
            env, env_params, noise_vector, args.num_eval_episodes, hp['episode_length']
        )
    
    # Pre-generate deterministic task sequence (same across methods for same trial)
    num_tasks = num_updates // task_interval
    task_rng = jax.random.key(trial * 7919)  # Separate RNG, deterministic per trial
    if task_type == 'noise':
        # Shared with source/studies/gymnax/continual_common.py, which post-hoc
        # evaluation also uses, so the sub-task an agent is scored on is the one
        # it was trained on -- and so a given trial faces the same sub-tasks
        # under every method.
        task_noise_vectors = make_task_noise_vectors(trial, num_tasks, obs_dim, noise_range)
        task_noise_vectors = cycle_task_sequence(task_noise_vectors, task_period)
        print(f"  Pre-generated {num_tasks} noise vectors (task 0 = zero noise)")
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
        # Alternating, NOT drawn from the trial seed -- see action_flip_sequence
        # for why a two-state regime must not be sampled. The observation is
        # untouched here, so the offset sequence is zeros, carried explicitly so
        # the saved artifacts have the same shape under all three families.
        task_flips = action_flip_sequence(num_tasks, task_period,
                                          env_name=env_name, trial=args.trial)
        task_noise_vectors = [jnp.zeros((obs_dim,))] * num_tasks
        print(f"  Pre-generated {num_tasks} action-reversal flags "
              f"(task 0 = stock order): {task_flips}")
    
    # Initialize noise vector for Task 0
    noise_vector = jnp.zeros((obs_dim,))
    current_task = 0
    
    if task_type == 'noise':
        noise_vector = task_noise_vectors[0]
        print(f"\n  Task 0 noise: {jax.device_get(noise_vector)}")
        print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")
    elif task_type == 'param':
        param_val = task_param_values[0]
        env_params = apply_physics(env_name, base_env_params, param_name, param_val)
        print(f"\n  Task 0 {param_name}: {param_val:.4f}x")
    elif task_type == 'actions':
        env_params = FlippedParams(base_env_params,
                                   jnp.float32(task_flips[0]))
        print(f"\n  Task 0 action order: "
              f"{'REVERSED' if task_flips[0] else 'stock'}")
    
    # Helper function to save GIFs for current task
    def save_task_gifs(task_idx, noise_vec, task_name):
        """Save evaluation GIFs at end of a task."""
        nonlocal key
        if args.no_gifs:
            return
        try:
            render_fn = get_render_fn(env_name)
            if render_fn is None:
                print(f"  Warning: No render function for {env_name}")
                return
            
            num_gifs = 10
            
            # Create task-specific subdirectory with task name
            task_gifs_dir = os.path.join(gifs_dir, f"task_{task_idx:02d}_{task_name}")
            os.makedirs(task_gifs_dir, exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            
            for gif_idx in range(num_gifs):
                key, gif_key = random.split(key)
                obs, env_state = env.reset(gif_key, env_params)
                
                frames = []
                total_reward = 0.0
                
                for step in range(hp['episode_length']):
                    # Render frame
                    frame = render_fn(obs, fig, ax, step=step)
                    frames.append(frame)
                    
                    # Get action (with noisy observation)
                    noisy_obs = obs + noise_vec
                    logits = policy_network.apply(policy_state.params, noisy_obs)
                    action = jnp.argmax(logits)
                    
                    # Step
                    gif_key, step_key = random.split(gif_key)
                    obs, env_state, reward, done, _ = env.step(step_key, env_state, action, env_params)
                    total_reward += float(reward)
                    
                    if bool(done):
                        break
                
                # Save GIF with task name in filename
                gif_path = os.path.join(task_gifs_dir, f"task{task_idx}_{task_name}_rollout_{gif_idx:02d}_reward{total_reward:.0f}.gif")
                imageio.mimsave(gif_path, frames, fps=30, loop=0)
            
            plt.close(fig)
            print(f"  Saved {num_gifs} GIFs for task {task_idx} ({task_name}) in {task_gifs_dir}")
            
        except Exception as e:
            print(f"  Warning: Failed to save GIFs for task {task_idx}: {e}")
    
    # The agent handed over at the end of each sub-task, flattened. This is what
    # source/studies/evaluate_continual.py scores; nothing here computes a
    # success rate.
    task_agents = []

    # Environments threaded across updates. None means "reset before the next
    # rollout" -- set at the start of the run here, and again at every sub-task
    # boundary, so no rollout ever spans two MDPs.
    env_carry = None

    # Metrics tracking
    solved_threshold = SOLVED_THRESHOLDS.get(env_name)
    task_start_update = 0
    # Separate from `task_start_update`, which is also the origin of the
    # `updates_to_threshold` DIAGNOSTIC and so must keep tracking boundaries.
    # This one is method state: it re-arms C-CHAIN's warmup gate, so it only
    # moves when the run was explicitly asked to give C-CHAIN the boundary.
    chain_task_start_update = 0
    task_updates_to_threshold = None
    all_metrics = []
    
    # Zero-shot eval for task 0: evaluate random init on task 0
    key, zt_key = random.split(key)
    task_zt_mean, task_zt_std = jit_evaluate(
        zt_key, policy_state.params, noise_vector, env_params
    )
    task_zt_mean = float(task_zt_mean)
    task_zt_std = float(task_zt_std)
    print(f"  Task 0 zero-shot: {task_zt_mean:.2f} +/- {task_zt_std:.2f}")
    
    # --- Plasticity diagnostics (source/metrics/plasticity.py) ------------------
    # Frozen probe batch, so churn and dormancy are measured on the SAME states
    # all run long and are comparable with the NE trainers and the stationary
    # block. The existing `dormant_frac` below is kept as-is -- it is scored on
    # each update's own batch, which answers a different question, and silently
    # redefining it would invalidate the runs that already report it.
    #
    # Private RNG stream; nothing here writes back into training.
    probe_key = random.key(seed + 3_000_000)
    probe_obs = None
    prev_policy_params = None
    _policy_act_fn = ACTIVATIONS[GYMNAX_POLICY_ACTIVATION]
    _value_act_fn = ACTIVATIONS[GYMNAX_VALUE_ACTIVATION]
    _policy_criterion = redo.criterion_for_activation(GYMNAX_POLICY_ACTIVATION)
    _value_criterion = redo.criterion_for_activation(GYMNAX_VALUE_ACTIVATION)

    def _policy_logits(params, obs_batch):
        return policy_network.apply(params, obs_batch)

    for update in range(num_updates):
        timestep = (update + 1) * timesteps_per_update
        
        # Check for task switch
        if update > 0 and update % task_interval == 0:
            # Save GIFs for the ending task BEFORE switching
            if task_type == 'noise':
                prev_noise_mag = float(jnp.linalg.norm(noise_vector))
                prev_task_name = f"noise_{prev_noise_mag:.2f}"
            elif task_type == 'param':
                prev_task_name = f"{param_name}_{task_param_values[current_task]:.4f}x"
            elif task_type == 'actions':
                prev_task_name = (
                    "actions_reversed" if task_flips[current_task] else "actions_stock")
            save_task_gifs(current_task, noise_vector, prev_task_name)
            
            # Per-task evaluation (10 trials) and checkpoint
            key, task_eval_key = random.split(key)
            task_eval_mean, task_eval_std = jit_evaluate(
                task_eval_key, policy_state.params, noise_vector, env_params
            )
            task_eval_mean = float(task_eval_mean)
            task_eval_std = float(task_eval_std)
            print(f"  Task {current_task} eval: {task_eval_mean:.2f} ± {task_eval_std:.2f}")
            task_ckpt_path = os.path.join(checkpoints_dir, f"task_{current_task}.pkl")
            ckpt_data = {
                'policy_params': policy_state.params,
                'value_params': value_state.params,
                'task_idx': current_task,
                'task_type': task_type,
                'update': update,
                'best_fitness': float(task_best_reward),
                'eval_mean': task_eval_mean,
                'eval_std': task_eval_std,
                'zero_shot_eval_mean': task_zt_mean,
                'zero_shot_eval_std': task_zt_std,
                'updates_to_threshold': task_updates_to_threshold,
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
            task_agents.append(np.asarray(ravel_pytree(policy_state.params)[0]))
            print(f"    Saved task checkpoint: {task_ckpt_path}")
            
            # Store per-task metrics
            all_metrics.append({
                'task_idx': current_task,
                'eval_mean': task_eval_mean,
                'eval_std': task_eval_std,
                'zero_shot_eval_mean': task_zt_mean,
                'zero_shot_eval_std': task_zt_std,
                'updates_to_threshold': task_updates_to_threshold,
            })
            
            task_best_reward = -float('inf')  # Reset for next task
            current_task += 1
            task_start_update = update
            task_updates_to_threshold = None
            if method == 'cchain' and cchain_reset_on_switch:
                # Only when explicitly asked for: this is a task-boundary signal
                # that no other method in the comparison receives. See the flag.
                chain_ctrl.reset()
                chain_task_start_update = update

            if reset_on_switch:
                # The plasticity upper bound: throw the agent away and start
                # the sub-task from a fresh initialisation, so nothing is
                # carried across the boundary. It cannot forget, because it
                # retains nothing, and it cannot transfer for the same reason
                # -- which is what makes it the reference the other methods are
                # read against rather than a method in its own right. C-CHAIN
                # reports this as its oracle (crl_run_ppo_dmc_oracle.py, which
                # re-creates agent and optimiser at every boundary).
                #
                # Done before the zero-shot evaluation below on purpose: that
                # evaluation is meant to describe the agent that is about to
                # train on this sub-task, and for this baseline that agent is
                # the fresh one. Resetting after would report the carried
                # agent's transfer and then discard the agent it measured.
                key, reset_p_key, reset_v_key = random.split(key, 3)
                policy_state = TrainState.create(
                    apply_fn=policy_network.apply,
                    params=policy_network.init(reset_p_key, dummy_obs),
                    tx=make_base_optimizer(),
                )
                value_state = TrainState.create(
                    apply_fn=value_network.apply,
                    params=value_network.init(reset_v_key, dummy_obs),
                    tx=make_base_optimizer(),
                )
                print(f"  [reset_on_switch] agent and optimiser re-initialised")
            if task_type == 'noise':
                noise_vector = task_noise_vectors[current_task]
                print(f"\n>>> Task {current_task} started at update {update}")
                print(f"  Noise vector: {jax.device_get(noise_vector)}")
                print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")
            elif task_type == 'param':
                param_val = task_param_values[current_task]
                env_params = apply_physics(env_name, base_env_params,
                                           param_name, param_val)
                print(f"\n>>> Task {current_task} started at update {update}")
                print(f"  {param_name}: {param_val:.4f}x")
            elif task_type == 'actions':
                flip = task_flips[current_task]
                env_params = FlippedParams(base_env_params, jnp.float32(flip))
                print(f"\n>>> Task {current_task} started at update {update}")
                print(f"  action order: {'REVERSED' if flip else 'stock'}")

            # Drop the environments the previous sub-task left mid-episode. The
            # noise vector or a physics parameter has just changed, so those
            # states belong to the old MDP; continuing them would put two MDPs'
            # transitions in one GAE trace and start this sub-task on data the
            # previous one generated.
            env_carry = None

            # Zero-shot evaluation on new task (before training)
            key, zt_key = random.split(key)
            task_zt_mean, task_zt_std = jit_evaluate(
                zt_key, policy_state.params, noise_vector, env_params
            )
            task_zt_mean = float(task_zt_mean)
            task_zt_std = float(task_zt_std)
            print(f"  Task {current_task} zero-shot: {task_zt_mean:.2f} +/- {task_zt_std:.2f}")
            wandb.summary[f"task_{current_task}_zero_shot_mean"] = task_zt_mean
            wandb.summary[f"task_{current_task}_zero_shot_std"] = task_zt_std
        
        # Collect rollout with current noise. env_carry threads the environments
        # across updates within a sub-task; it was set to None just above if this
        # update crossed a boundary, so the new sub-task starts from reset.
        if env_carry is None:
            key, carry_key = random.split(key)
            env_carry = jit_init_env_carry(carry_key, env_params)

        key, rollout_key = random.split(key)
        rollout, env_carry, last_obs = jit_collect_rollout_fn(
            rollout_key, policy_state.params, value_state.params, noise_vector,
            env_params, env_carry
        )

        last_values = value_network.apply(value_state.params, last_obs)

        # Freeze the probe states on the first rollout of the run. They stay
        # fixed across sub-task boundaries on purpose: churn is meant to measure
        # the policy moving, not the observation distribution moving under it.
        if probe_obs is None:
            probe_obs = collect_probe_states(
                rollout['obs'], num_probe=args.num_probe_states, key=probe_key)
        prev_policy_params = policy_state.params

        # Compute advantages and returns (JIT-compiled vectorized GAE)
        all_advantages, all_returns = compute_advantages_returns(
            rollout['rewards'], rollout['values'], rollout['dones'], last_values
        )
        
        # Flatten rollout for training
        batch_size = hp['num_steps'] * hp['num_envs']
        flat_batch = {
            'obs': rollout['obs'].reshape(batch_size, -1),
            'actions': rollout['actions'].reshape(batch_size),
            'log_probs': rollout['log_probs'].reshape(batch_size),
            'advantages': all_advantages.reshape(batch_size),
            'returns': all_returns.reshape(batch_size),
        }
        
        # Training epochs (compiled as jax.lax.scan for speed)
        key, sgd_key = random.split(key)
        if method == 'cchain':
            policy_state, value_state, chain_state, loss, metrics = jit_sgd_epochs_chain(
                policy_state, value_state, flat_batch, ent_coef,
                chain_ctrl.coef, chain_state, sgd_key
            )
            # Auto-tune the churn coefficient from the relative loss scales,
            # once enough updates into the current task.
            # Was `update - task_start_update`, which re-armed the warmup
            # gate at every boundary even with --cchain_reset_on_switch off,
            # so the coefficient froze for `chain_warmup_updates` after every
            # switch. That was a boundary signal the flag was meant to
            # withhold. Fixed 2026-09-08; with the flag off this is the
            # global update index.
            chain_ctrl.update(metrics['chain_p_loss'], metrics['chain_p_reg_loss'],
                              update - chain_task_start_update)
        else:
            policy_state, value_state, joint_opt_state, loss, metrics = jit_sgd_epochs(
                policy_state, value_state, joint_opt_state, flat_batch, ent_coef, sgd_key
            )

        # TRAC needs no per-update hook: the tuner lives inside the optimizer
        # and has already run, once per gradient step, inside the scan above.

        # ReDo: recycle dormant neurons
        if method == 'redo' and (update + 1) % args.redo_interval == 0:
            key, redo_key = random.split(key)
            redo_obs = flat_batch['obs'][:min(args.redo_batch_size, flat_batch['obs'].shape[0])]
            policy_state, value_state, redo_stats = run_redo_pass(
                policy_state, value_state, redo_obs, redo_key, args, hp
            )
            for net_name, st in redo_stats.items():
                wandb.log({
                    'timestep': timestep,
                    f'redo/{net_name}_dormant_count': st['dormant_count'],
                    f'redo/{net_name}_dormant_fraction': st['dormant_fraction'],
                    f'redo/{net_name}_zero_fraction': st['zero_fraction'],
                })

        # CBP: Update utility and replace low-utility neurons
        num_replaced = 0
        if method == 'cbp':
            key, cbp_key = random.split(key)
            sample_obs = flat_batch['obs'][:min(1024, flat_batch['obs'].shape[0])]
            policy_state, cbp_tracker, num_replaced = apply_cbp(
                policy_state, cbp_tracker, sample_obs, cbp_key,
                args.cbp_replacement_rate, args.cbp_maturity_threshold,
            )
        
        # Evaluate
        if (update + 1) % args.eval_interval == 0 or update == num_updates - 1:
            key, eval_key = random.split(key)
            mean_reward, std_reward = jit_evaluate(
                eval_key, policy_state.params, noise_vector, env_params
            )
            mean_reward = float(mean_reward)
            std_reward = float(std_reward)
            
            if mean_reward > best_reward:
                best_reward = mean_reward
            if mean_reward > task_best_reward:
                task_best_reward = mean_reward
            
            # Track dormant neurons. Both heads are measured: ReDo recycles
            # them separately (--redo_targets), and the value net is the one
            # that goes dormant first under PPO, so a policy-only count
            # understates the plasticity loss.
            dormant_sample = flat_batch['obs'][:min(1024, flat_batch['obs'].shape[0])]
            dormant_per_layer, total_dormant, total_neurons = count_dormant_neurons(
                policy_state.params, dormant_sample, hp['policy_hidden_dims'],
                tau=args.redo_tau,
            )
            dormant_frac = total_dormant / max(total_neurons, 1)
            v_dormant_per_layer, v_total_dormant, v_total_neurons = count_dormant_neurons(
                value_state.params, dormant_sample, hp['value_hidden_dims'],
                tau=args.redo_tau,
            )
            v_dormant_frac = v_total_dormant / max(v_total_neurons, 1)

            # Comparable columns: same frozen probe states, same criterion, same
            # churn definition as every other method including the NE trainers.
            probe_churn = plasticity.churn_action_disagreement(
                _policy_logits, prev_policy_params, policy_state.params, probe_obs)

            # C-CHAIN's OWN churn estimator, now computed for EVERY method and not
            # just cchain. The reference does exactly this: on discrete control it
            # reports the cross-entropy H(pi_before, pi_after)
            # (crl_procgen/vis_train_procgen_c_chain.py:192), and its VANILLA PPO
            # script logs the identical quantity -- churn-for-all-methods is the
            # published practice, not an extension of it.
            #
            # The delta is one PPO UPDATE, matching the action-disagreement column
            # above so the two gymnax churn columns share a clock. C-CHAIN's internal
            # control signal is per-GRADIENT-STEP and is still logged separately as
            # chain/policy_churn; do not merge the two.
            _churn_ce = float(jnp.mean(chain_policy_churn(
                _policy_logits(prev_policy_params, probe_obs),
                _policy_logits(policy_state.params, probe_obs))))

            # NTK effective rank -- C-CHAIN's own plasticity indicator, and the CAUSE
            # its argument assigns to churn (rank collapse -> correlated gradients ->
            # churn). Same frozen probe batch as churn and dormancy, so all three
            # describe the same network on the same states. See
            # source/metrics/ntk.py.
            _ntk = ntk_metrics.ntk_rank_stats(
                lambda p, o: policy_network.apply(p, o), policy_state.params, probe_obs,
                prefix='policy_ntk')
            probe_p_dormant = redo.dormant_stats(
                policy_state.params, probe_obs, len(hp['policy_hidden_dims']),
                args.redo_tau, activation_fn=_policy_act_fn,
                criterion=_policy_criterion)['dormant_fraction']
            probe_v_dormant = redo.dormant_stats(
                value_state.params, probe_obs, len(hp['value_hidden_dims']),
                args.redo_tau, activation_fn=_value_act_fn,
                criterion=_value_criterion)['dormant_fraction']

            # Track updates to threshold (SU metric)
            if solved_threshold is not None and task_updates_to_threshold is None:
                if mean_reward >= solved_threshold:
                    task_updates_to_threshold = (update + 1) - task_start_update
                    print(f"  Task {current_task} reached threshold {solved_threshold} at update {update+1} (updates_in_task={task_updates_to_threshold})")
            
            elapsed = time.time() - start_time
            
            # Weight statistics, the third plasticity signal alongside dormancy
            # and churn -- the failure those two miss is a norm that grows
            # without bound across the sub-task sequence until the network is
            # hard to move while every unit is still nominally active. Policy
            # and value separately: they have different parameter counts, so
            # only the `_rms` columns are comparable between them.
            _wstats = {
                **weight_stats(policy_state.params, prefix='policy_weight'),
                **weight_stats(value_state.params, prefix='value_weight'),
            }

            training_metrics.append({
                'timestep': timestep,
                'update': update + 1,
                'task': current_task,
                **_wstats,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'best_reward': best_reward,
                'entropy': float(metrics['entropy']),
                'pg_loss': float(metrics['pg_loss']),
                'vf_loss': float(metrics['vf_loss']),
                'ent_coef': args.ent_coef,
                'noise_magnitude': float(jnp.linalg.norm(noise_vector)) if task_type == 'noise' else 0.0,
                'param_mult': float(task_param_values[current_task]) if task_type == 'param' else None,
                'action_flip': int(task_flips[current_task]) if task_type == 'actions' else None,
                'elapsed_time': elapsed,
                # 'dormant_neurons'/'dormant_frac' are the policy net, kept
                # under their original names so older runs stay readable.
                'dormant_neurons': total_dormant,
                'dormant_frac': dormant_frac,
                'policy_churn_action': probe_churn,
                'policy_dormant_fraction_probe': probe_p_dormant,
                'value_dormant_fraction_probe': probe_v_dormant,
                'policy_dormant_neurons': total_dormant,
                'policy_dormant_frac': dormant_frac,
                'policy_total_neurons': total_neurons,
                'value_dormant_neurons': v_total_dormant,
                'value_dormant_frac': v_dormant_frac,
                'value_total_neurons': v_total_neurons,
                'chain_p_reg_coef': chain_ctrl.coef if method == 'cchain' else None,
                'chain_p_reg_loss': float(metrics['chain_p_reg_loss']) if method == 'cchain' else None,
                'policy_churn': _churn_ce,
                **_ntk,
                'value_churn': float(metrics['value_churn']) if method == 'cchain' else None,
            })
            
            log_dict = {
                'timestep': timestep,
                'task': current_task,
                'eval/mean_reward': mean_reward,
                'eval/best_reward': best_reward,
                'train/entropy': metrics['entropy'],
                'train/pg_loss': metrics['pg_loss'],
                'train/vf_loss': metrics['vf_loss'],
                'train/ent_coef': args.ent_coef,
                'dormant/total': total_dormant,
                'dormant/fraction': dormant_frac,
                'plasticity/policy_churn_action': probe_churn,
                'plasticity/policy_dormant_fraction': probe_p_dormant,
                'plasticity/value_dormant_fraction': probe_v_dormant,
                'dormant/value_total': v_total_dormant,
                'dormant/value_fraction': v_dormant_frac,
            }
            for li, (n_d, l_sz) in enumerate(dormant_per_layer):
                log_dict[f'dormant/layer_{li}'] = n_d
            for li, (n_d, l_sz) in enumerate(v_dormant_per_layer):
                log_dict[f'dormant/value_layer_{li}'] = n_d
            if method == 'cbp':
                log_dict['cbp/neurons_replaced'] = num_replaced
            if method == 'cchain':
                log_dict['chain/p_reg_coef'] = chain_ctrl.coef
                log_dict['chain/p_reg_loss'] = float(metrics['chain_p_reg_loss'])
                log_dict['chain/policy_churn'] = float(metrics['policy_churn'])
                log_dict['chain/value_churn'] = float(metrics['value_churn'])
            if task_type == 'noise':
                log_dict['noise_magnitude'] = float(jnp.linalg.norm(noise_vector))
            elif task_type == 'param':
                log_dict[f"{param_name}_mult"] = float(task_param_values[current_task])
            elif task_type == 'actions':
                log_dict['action_flip'] = int(task_flips[current_task])
            wandb.log(log_dict)
            
            dormant_layer_str = ' '.join(f'L{i}:{nd}/{ls}' for i, (nd, ls) in enumerate(dormant_per_layer))
            print(f"Update {update+1:5d} | Task {current_task} | Timestep {timestep:10,} | "
                  f"Reward: {mean_reward:8.2f} ± {std_reward:.2f} | Best: {best_reward:8.2f} | "
                  f"Dormant pi: {total_dormant}/{total_neurons} ({dormant_frac:.1%}) [{dormant_layer_str}] | "
                  f"V: {v_total_dormant}/{v_total_neurons} ({v_dormant_frac:.1%})")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s, Best (training): {best_reward:.2f}")
    
    # Final evaluation with 10 trials
    if task_type == 'noise':
        print(f"\nFinal evaluation (10 trials) on Task {current_task} with noise magnitude {float(jnp.linalg.norm(noise_vector)):.4f}...")
    elif task_type == 'param':
        print(f"\nFinal evaluation (10 trials) on Task {current_task} with "
              f"{param_name} at {task_param_values[current_task]:.4f}x...")
    elif task_type == 'actions':
        print(f"\nFinal evaluation (10 trials) on Task {current_task} with "
              f"action order "
              f"{'REVERSED' if task_flips[current_task] else 'stock'}...")
    key, final_eval_key = random.split(key)
    final_mean, final_std = jit_evaluate(
        final_eval_key, policy_state.params, noise_vector, env_params
    )
    final_mean = float(final_mean)
    final_std = float(final_std)
    
    print(f"\nFinal evaluation results:")
    print(f"  Mean: {final_mean:.2f} +/- {final_std:.2f}")
    print(f"  Training best: {best_reward:.2f}")
    
    wandb.log({
        "final_eval_mean": final_mean,
        "final_eval_std": final_std,
    })
    
    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"{method}_{env_name.replace('-', '_')}_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'policy_params': policy_state.params,
            'value_params': value_state.params,
            'best_reward': best_reward,
            'final_eval_mean': final_mean,
            'final_eval_std': final_std,
            'config': config,
        }, f)
    print(f"Saved: {ckpt_path}")
    
    # Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    # Save GIFs for the final task
    if task_type == 'noise':
        final_noise_mag = float(jnp.linalg.norm(noise_vector))
        final_task_name = f"noise_{final_noise_mag:.2f}"
    elif task_type == 'param':
        final_task_name = f"{param_name}_{task_param_values[current_task]:.4f}x"
    elif task_type == 'actions':
        final_task_name = (
            "actions_reversed" if task_flips[current_task] else "actions_stock")
    save_task_gifs(current_task, noise_vector, final_task_name)
    
    # Save final task checkpoint
    task_ckpt_path = os.path.join(checkpoints_dir, f"task_{current_task}.pkl")
    final_ckpt_data = {
        'policy_params': policy_state.params,
        'value_params': value_state.params,
        'task_idx': current_task,
        'task_type': task_type,
        'update': num_updates,
        'best_fitness': float(task_best_reward),
        'eval_mean': final_mean,
        'eval_std': final_std,
        'zero_shot_eval_mean': task_zt_mean,
        'zero_shot_eval_std': task_zt_std,
        'updates_to_threshold': task_updates_to_threshold,
    }
    if task_type == 'noise':
        final_ckpt_data['noise_vector'] = jax.device_get(noise_vector)
    elif task_type == 'param':
        final_ckpt_data['param_name'] = param_name
        final_ckpt_data['param_mult'] = float(task_param_values[current_task])
    with open(task_ckpt_path, 'wb') as f:
        pickle.dump(final_ckpt_data, f)
    task_agents.append(np.asarray(ravel_pytree(policy_state.params)[0]))
    print(f"Saved final task checkpoint: {task_ckpt_path}")
    
    # Store final task metrics
    all_metrics.append({
        'task_idx': current_task,
        'eval_mean': final_mean,
        'eval_std': final_std,
        'zero_shot_eval_mean': task_zt_mean,
        'zero_shot_eval_std': task_zt_std,
        'updates_to_threshold': task_updates_to_threshold,
    })
    
    # Compute aggregate metrics and save to YAML
    num_tasks = len(all_metrics)
    num_solved = sum(1 for m in all_metrics if solved_threshold is not None and m['eval_mean'] >= solved_threshold)
    success_rate = num_solved / num_tasks if num_tasks > 0 else 0.0
    zt_values = [m['zero_shot_eval_mean'] for m in all_metrics]
    
    summary_metrics = {
        'method': method,
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
    metrics_yaml_path = os.path.join(output_dir, "metrics.yaml")
    with open(metrics_yaml_path, 'w') as f:
        yaml.dump(summary_metrics, f, default_flow_style=False)
    print(f"Saved metrics: {metrics_yaml_path}")

    # Artifacts for post-hoc evaluation (source/studies/evaluate_continual.py).
    # The success_rate in metrics.yaml above comes from the small evaluation
    # done during training; the numbers to report are the ones the evaluator
    # computes from these agents at a controlled episode count.
    #
    # EMITTED UNDER BOTH TASK TYPES since 2026-09-08 -- see the same note in
    # the GA trainer. A param run used to write no results.json at all, so it
    # was invisible to the evaluator AND to verify_runs.py.
    if True:
        agents = np.stack(task_agents)
        arrays = dict(
            final=agents,
            noise_vectors=np.stack([np.asarray(v) for v in task_noise_vectors[:len(agents)]]),
        )
        if task_type == 'param':
            arrays['param_mults'] = np.asarray(
                task_param_values[:len(agents)], dtype=np.float64)
        if task_type == 'actions':
            arrays['action_flips'] = np.asarray(
                task_flips[:len(agents)], dtype=np.int32)
        np.savez_compressed(os.path.join(output_dir, "checkpoints.npz"), **arrays)
        results = {
            'method': method,
            'env': env_name,
            'trial': trial,
            'seed': seed,
            'pop_size': None,  # gradient method: one agent, no population
            'num_tasks': int(agents.shape[0]),
            'task_interval': task_interval,
            'num_timesteps': num_timesteps,
            'episode_length': hp['episode_length'],
            'hidden_dims': list(hp['policy_hidden_dims']),
            'num_params': int(agents.shape[1]),
            'noise_range': noise_range,
            'task_type': task_type,
            'param_name': param_name if task_type == 'param' else None,
            'action_flips': ([int(f) for f in task_flips[:len(agents)]]
                             if task_type == 'actions' else None),
            # One agent per sub-task, so there is no finalgen/incumbent
            # distinction to make; the evaluator reads this list.
            'agent_sources': ['final'],
            'elapsed_seconds': total_time,
            'config': config,
            'per_task': all_metrics,
        }
        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved {agents.shape[0]} sub-task agents to checkpoints.npz; "
              f"score them with source/studies/evaluate_continual.py")

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
