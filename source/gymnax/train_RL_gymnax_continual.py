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
- TRAC: Trust Region Actor-Critic (automatic entropy tuning)
- ReDo: Reinitializing Dormant Neurons

Usage:
    python train_RL_gymnax_continual.py --env CartPole-v1 --method ppo --gpus 0
    python train_RL_gymnax_continual.py --env Acrobot-v1 --method trac --gpus 0
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
matplotlib.use('Agg')  # Headless backend for GIF rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.lines import Line2D


# Solved thresholds for Speed-Up (SU) metric
SOLVED_THRESHOLDS = {
    'CartPole-v1': 475,
    'Acrobot-v1': -70,
    'MountainCar-v0': -110,
}

# ============================================================================
# Logging Helper
# ============================================================================

class Tee:
    """Duplicate stdout to a file and console."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w', buffering=1)
        self.stdout = sys.stdout
    
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        self.file.close()


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

class PolicyNetwork(nn.Module):
    """Discrete policy network with categorical output."""
    hidden_dims: tuple = (16, 16)
    action_dim: int = 2
    
    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits


class ValueNetwork(nn.Module):
    """Value network for state value estimation."""
    hidden_dims: tuple = (128, 128, 128)
    
    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        value = nn.Dense(1)(x)
        return value.squeeze(-1)


# ============================================================================
# PPO Functions
# ============================================================================

def categorical_sample(key, logits):
    """Sample from categorical distribution."""
    return jax.random.categorical(key, logits)


def categorical_log_prob(logits, action):
    """Log probability of action under categorical distribution."""
    log_probs = jax.nn.log_softmax(logits)
    return log_probs[action]


def categorical_entropy(logits):
    """Entropy of categorical distribution."""
    log_probs = jax.nn.log_softmax(logits)
    probs = jax.nn.softmax(logits)
    return -jnp.sum(probs * log_probs, axis=-1)


def gae_advantages(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute GAE advantages using jax.lax.scan (GPU-friendly)."""
    # Append zero for bootstrapping
    values_with_bootstrap = jnp.concatenate([values, jnp.array([0.0])])
    
    def scan_fn(gae, t):
        # t goes from 0 to T-1 (reversed order via [::-1])
        delta = rewards[t] + gamma * values_with_bootstrap[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        return gae, gae
    
    # Scan in reverse order
    indices = jnp.arange(len(rewards))[::-1]
    _, advantages_reversed = jax.lax.scan(scan_fn, 0.0, indices)
    
    return advantages_reversed[::-1]


# ============================================================================
# Vectorized Environment Step (for parallel rollouts)
# ============================================================================

def make_vec_env_fns(env, env_params, num_envs):
    """Create vectorized reset and step functions."""
    
    def vec_reset(key):
        keys = random.split(key, num_envs)
        return jax.vmap(lambda k: env.reset(k, env_params))(keys)
    
    def vec_step(keys, states, actions):
        return jax.vmap(lambda k, s, a: env.step(k, s, a, env_params))(keys, states, actions)
    
    return jax.jit(vec_reset), jax.jit(vec_step)


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
):
    """Collect rollout data from parallel environments using jax.lax.scan (GPU-friendly).
    
    Args:
        noise_vector: Observation noise for continual learning. Shape: (obs_dim,)
    """
    
    # Initialize environments
    key, reset_key = random.split(key)
    reset_keys = random.split(reset_key, num_envs)
    obs, states = jax.vmap(lambda k: env.reset(k, env_params))(reset_keys)
    
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
            lambda fresh, old: jnp.where(dones[:, None] if fresh.ndim > 1 else dones, fresh, old),
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
    _, rollout = jax.lax.scan(env_step, (obs, states, key), None, length=num_steps)
    
    # rollout is a dict with arrays of shape (num_steps, num_envs, ...)
    return rollout


def compute_ppo_loss(
    policy_params,
    value_params,
    policy_network,
    value_network,
    batch,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
):
    """Compute PPO loss."""
    obs = batch['obs']
    actions = batch['actions']
    old_log_probs = batch['log_probs']
    advantages = batch['advantages']
    returns = batch['returns']
    
    # Forward pass
    logits = policy_network.apply(policy_params, obs)
    values = value_network.apply(value_params, obs)
    
    # Policy loss
    log_probs = jax.vmap(categorical_log_prob)(logits, actions)
    ratio = jnp.exp(log_probs - old_log_probs)
    
    # Clipped surrogate objective
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
    
    # Value loss
    vf_loss = 0.5 * jnp.square(values - returns).mean()
    
    # Entropy bonus
    entropy = jax.vmap(categorical_entropy)(logits).mean()
    
    # Total loss
    loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy
    
    return loss, {
        'pg_loss': pg_loss,
        'vf_loss': vf_loss,
        'entropy': entropy,
        'approx_kl': jnp.mean((ratio - 1) - jnp.log(ratio)),
    }


def train_step(
    policy_network,
    value_network,
    clip_eps,
    vf_coef,
    policy_state,
    value_state,
    batch,
    ent_coef,
):
    """Perform one training step."""
    
    def loss_fn(policy_params, value_params):
        return compute_ppo_loss(
            policy_params, value_params,
            policy_network, value_network,
            batch, clip_eps, vf_coef, ent_coef,
        )
    
    # Compute gradients
    (loss, metrics), (policy_grads, value_grads) = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(policy_state.params, value_state.params)
    
    # Update
    policy_state = policy_state.apply_gradients(grads=policy_grads)
    value_state = value_state.apply_gradients(grads=value_grads)
    
    return policy_state, value_state, loss, metrics


# ============================================================================
# ReDo: Reinitializing Dormant Neurons
# ============================================================================

def detect_dormant_neurons(activations, tau=0.01):
    """Detect dormant neurons based on activation statistics."""
    # activations: (batch, neurons)
    mean_activation = jnp.abs(activations).mean(axis=0)
    max_activation = jnp.max(mean_activation)
    dormant_mask = mean_activation < tau * max_activation
    return dormant_mask


def apply_redo_to_network(params, key, layer_name='Dense_0', tau=0.01):
    """
    Simple ReDo implementation - reinitialize dormant neurons.
    This is a simplified version that works with our small networks.
    """
    # For simplicity, we just add small noise to all weights periodically
    # A full implementation would track activations and reinit dormant neurons
    noise_scale = 0.001
    
    def add_noise(k, p):
        if p.ndim >= 2:  # Only add noise to weight matrices
            return p + noise_scale * random.normal(k, p.shape)
        return p
    
    flat_params = jax.tree_util.tree_leaves(params)
    keys = random.split(key, len(flat_params))
    new_flat = [add_noise(k, p) for k, p in zip(keys, flat_params)]
    
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params), new_flat)


# ============================================================================
# TRAC: Automatic Entropy Coefficient Tuning
# ============================================================================

def update_entropy_coef(ent_coef, target_entropy, current_entropy, lr=0.01):
    """Update entropy coefficient for TRAC."""
    # Increase ent_coef if entropy is below target, decrease if above
    log_ent_coef = jnp.log(ent_coef + 1e-8)
    log_ent_coef = log_ent_coef - lr * (current_entropy - target_entropy)
    return jnp.clip(jnp.exp(log_ent_coef), 1e-4, 1.0)


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
        "normalize_observations": True,
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
        "normalize_observations": True,
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
        "gamma": 0.95,  # discounting
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
    parser = argparse.ArgumentParser(description='PPO on Gymnax (Continual Learning)')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'])
    parser.add_argument('--method', type=str, default='ppo',
                        choices=['ppo', 'trac', 'redo'])
    parser.add_argument('--num_timesteps', type=int, default=None,
                        help='Override default timesteps for env')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)
    
    # Continual learning parameters
    parser.add_argument('--task_interval', type=int, default=None,
                        help='Change task every N updates (default: env-specific for 10 tasks)')
    parser.add_argument('--noise_range', type=float, default=1.0,
                        help='Scale for observation noise (task definition)')
    parser.add_argument('--task_type', type=str, default='noise',
                        choices=['noise', 'param'],
                        help='Type of task variation: noise (obs noise) or param (vary env parameter)')
    parser.add_argument('--param_range', type=float, nargs=2, default=None,
                        help='Range [min, max] for param sampling (task_type=param). Default: env-specific.')
    
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
    
    # TRAC specific
    parser.add_argument('--trac_target_entropy', type=float, default=None,
                        help='Target entropy for TRAC (default: 0.5 * log(action_dim))')
    
    # ReDo specific
    parser.add_argument('--redo_interval', type=int, default=50,
                        help='Apply ReDo every N updates')
    parser.add_argument('--redo_tau', type=float, default=0.01)
    
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
    
    # Environment-specific parameter variation config
    PARAM_CONFIGS = {
        'CartPole-v1': {'param_name': 'gravity', 'default_val': 9.8, 'default_range': [0.098, 198.0]},
        'MountainCar-v0': {'param_name': 'gravity', 'default_val': 0.0025, 'default_range': [0.00125, 0.005]},
        'Acrobot-v1': {'param_name': 'link_length_1', 'default_val': 1.0, 'default_range': [0.5, 2.0]},
    }
    param_cfg = PARAM_CONFIGS[env_name]
    param_name = param_cfg['param_name']
    param_range = args.param_range if args.param_range is not None else param_cfg['default_range']
    
    # Get environment-specific hyperparameters
    hp = get_hyperparams(args)
    num_timesteps = hp['num_timesteps']
    task_interval = hp['task_interval']
    
    print("=" * 60)
    print(f"PPO ({method.upper()}) on {env_name} (CONTINUAL)")
    print("=" * 60)
    
    # Create environment
    env, env_params = gymnax.make(env_name)
    env_params = env_params.replace(max_steps_in_episode=hp['episode_length'])
    obs_dim = env.observation_space(env_params).shape[0]
    action_dim = env.action_space(env_params).n
    
    print(f"  Environment: {env_name}")
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"  Method: {method.upper()}")
    print(f"  Total timesteps: {num_timesteps:,}")
    print(f"  Task interval: {task_interval} updates")
    if task_type == 'noise':
        print(f"  Task type: noise, range: {noise_range}")
    elif task_type == 'param':
        print(f"  Task type: param ({param_name}), range: {param_range}")
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
    
    # Create optimizers
    policy_optimizer = optax.adam(hp['learning_rate'])
    value_optimizer = optax.adam(hp['learning_rate'])
    
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
    
    # Initialize wandb
    config = {
        'env': env_name, 'method': method, 'num_timesteps': num_timesteps,
        'seed': seed, 'trial': trial, 'num_envs': hp['num_envs'],
        'num_steps': hp['num_steps'], 'learning_rate': hp['learning_rate'],
        'gamma': hp['gamma'], 'ent_coef': hp['ent_coef'],
        'num_epochs': hp['num_epochs'], 'num_minibatches': hp['num_minibatches'],
        'policy_hidden_dims': hp['policy_hidden_dims'], 'value_hidden_dims': hp['value_hidden_dims'],
        'task_interval': task_interval, 'task_type': task_type,
        'noise_range': noise_range, 'param_name': param_name, 'param_range': param_range,
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"{method}_{env_name}_continual_{task_type}_epochs{hp['num_epochs']}_trial{trial}", reinit=True)
    
    # TRAC: Initialize entropy coefficient
    ent_coef = hp['ent_coef']
    if method == 'trac':
        target_entropy = args.trac_target_entropy
        if target_entropy is None:
            target_entropy = 0.5 * jnp.log(action_dim)
        print(f"  TRAC target entropy: {target_entropy:.4f}")
    
    # Training metrics
    best_reward = -float('inf')
    task_best_reward = -float('inf')
    training_metrics = []
    start_time = time.time()
    
    # Calculate number of updates
    timesteps_per_update = hp['num_envs'] * hp['num_steps']
    num_updates = num_timesteps // timesteps_per_update
    
    print(f"\nStarting continual training ({num_updates} updates)...")
    
    # JIT compile rollout collection (includes noise_vector and env_params)
    @jax.jit
    def jit_collect_rollout_fn(key, policy_params, value_params, noise_vector, env_params):
        return collect_rollout(
            key, policy_network, policy_params,
            value_network, value_params,
            env, env_params, hp['num_envs'], hp['num_steps'],
            noise_vector
        )
    
    # JIT compile SGD epochs (replaces Python for-loops over epochs/minibatches)
    num_minibatches = hp['num_minibatches']
    num_epochs = hp['num_epochs']
    batch_size = hp['num_steps'] * hp['num_envs']
    minibatch_size = batch_size // num_minibatches
    
    @jax.jit
    def jit_sgd_epochs(policy_state, value_state, flat_batch, ent_coef, key):
        """Run num_epochs of SGD with num_minibatches each, fully compiled."""
        
        def single_epoch(carry, _):
            policy_state, value_state, key = carry
            key, shuffle_key = random.split(key)
            perm = random.permutation(shuffle_key, batch_size)
            shuffled = {k: v[perm] for k, v in flat_batch.items()}
            
            # Reshape into (num_minibatches, minibatch_size, ...)
            mb_data = {
                k: v.reshape(num_minibatches, minibatch_size, *v.shape[1:])
                for k, v in shuffled.items()
            }
            
            def single_minibatch(carry, minibatch):
                policy_state, value_state = carry
                policy_state, value_state, loss, metrics = train_step(
                    policy_network, value_network,
                    hp['clip_eps'], hp['vf_coef'],
                    policy_state, value_state, minibatch, ent_coef,
                )
                return (policy_state, value_state), metrics
            
            (policy_state, value_state), all_metrics = jax.lax.scan(
                single_minibatch, (policy_state, value_state), mb_data,
                length=num_minibatches
            )
            
            return (policy_state, value_state, key), all_metrics
        
        (policy_state, value_state, key), epoch_metrics = jax.lax.scan(
            single_epoch, (policy_state, value_state, key), None,
            length=num_epochs
        )
        
        # Return last metrics (last epoch, last minibatch)
        last_metrics = jax.tree_util.tree_map(lambda x: x[-1, -1], epoch_metrics)
        last_loss = last_metrics.get('pg_loss', 0.0) + hp['vf_coef'] * last_metrics.get('vf_loss', 0.0)
        
        return policy_state, value_state, last_loss, last_metrics
    
    # JIT compile vectorized GAE computation
    @jax.jit
    def compute_advantages_returns(rewards, values, dones):
        """Vectorized GAE computation over all environments."""
        # Input shapes: (num_steps, num_envs)
        rewards_T = rewards.T  # (num_envs, num_steps)
        values_T = values.T
        dones_T = dones.T
        
        advantages_T = jax.vmap(
            lambda r, v, d: gae_advantages(r, v, d, hp['gamma'], hp['gae_lambda'])
        )(rewards_T, values_T, dones_T)
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
        task_noise_vectors = [jnp.zeros((obs_dim,))]  # Task 0 = no noise (identical to non-continual)
        for t in range(1, num_tasks):
            task_rng, noise_key = random.split(task_rng)
            nv = random.normal(noise_key, (obs_dim,)) * noise_range
            task_noise_vectors.append(nv)
        print(f"  Pre-generated {num_tasks} noise vectors (task 0 = zero noise)")
    elif task_type == 'param':
        task_param_values = [param_cfg['default_val']]  # Task 0 = default
        for t in range(1, num_tasks):
            task_rng, param_key = random.split(task_rng)
            pv = float(random.uniform(param_key, minval=param_range[0], maxval=param_range[1]))
            task_param_values.append(pv)
        print(f"  Pre-generated {num_tasks} param values: {task_param_values}")
    
    # Initialize noise vector for Task 0
    noise_vector = jnp.zeros((obs_dim,))
    current_task = 0
    
    if task_type == 'noise':
        noise_vector = task_noise_vectors[0]
        print(f"\n  Task 0 noise: {jax.device_get(noise_vector)}")
        print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")
    elif task_type == 'param':
        param_val = task_param_values[0]
        env_params = env_params.replace(**{param_name: param_val})
        print(f"\n  Task 0 {param_name}: {param_val:.4f}")
    
    # Helper function to save GIFs for current task
    def save_task_gifs(task_idx, noise_vec, task_name):
        """Save evaluation GIFs at end of a task."""
        nonlocal key
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
    
    # Metrics tracking
    solved_threshold = SOLVED_THRESHOLDS.get(env_name)
    task_start_update = 0
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
    
    for update in range(num_updates):
        timestep = (update + 1) * timesteps_per_update
        
        # Check for task switch
        if update > 0 and update % task_interval == 0:
            # Save GIFs for the ending task BEFORE switching
            if task_type == 'noise':
                prev_noise_mag = float(jnp.linalg.norm(noise_vector))
                prev_task_name = f"noise_{prev_noise_mag:.2f}"
            elif task_type == 'param':
                prev_task_name = f"{param_name}_{float(getattr(env_params, param_name)):.4f}"
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
                ckpt_data['param_value'] = float(getattr(env_params, param_name))
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
                'updates_to_threshold': task_updates_to_threshold,
            })
            
            task_best_reward = -float('inf')  # Reset for next task
            current_task += 1
            task_start_update = update
            task_updates_to_threshold = None
            if task_type == 'noise':
                noise_vector = task_noise_vectors[current_task]
                print(f"\n>>> Task {current_task} started at update {update}")
                print(f"  Noise vector: {jax.device_get(noise_vector)}")
                print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")
            elif task_type == 'param':
                param_val = task_param_values[current_task]
                env_params = env_params.replace(**{param_name: param_val})
                print(f"\n>>> Task {current_task} started at update {update}")
                print(f"  {param_name}: {param_val:.4f}")
            
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
        
        # Collect rollout with current noise
        key, rollout_key = random.split(key)
        rollout = jit_collect_rollout_fn(
            rollout_key, policy_state.params, value_state.params, noise_vector, env_params
        )
        
        # Compute advantages and returns (JIT-compiled vectorized GAE)
        all_advantages, all_returns = compute_advantages_returns(
            rollout['rewards'], rollout['values'], rollout['dones']
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
        policy_state, value_state, loss, metrics = jit_sgd_epochs(
            policy_state, value_state, flat_batch, ent_coef, sgd_key
        )
        
        # TRAC: Update entropy coefficient
        if method == 'trac':
            ent_coef = update_entropy_coef(
                ent_coef, target_entropy, metrics['entropy']
            )
        
        # ReDo: Reinitialize dormant neurons
        if method == 'redo' and (update + 1) % args.redo_interval == 0:
            key, redo_key = random.split(key)
            policy_state = policy_state.replace(
                params=apply_redo_to_network(policy_state.params, redo_key, tau=args.redo_tau)
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
            
            # Track updates to threshold (SU metric)
            if solved_threshold is not None and task_updates_to_threshold is None:
                if mean_reward >= solved_threshold:
                    task_updates_to_threshold = (update + 1) - task_start_update
                    print(f"  Task {current_task} reached threshold {solved_threshold} at update {update+1} (updates_in_task={task_updates_to_threshold})")
            
            elapsed = time.time() - start_time
            
            training_metrics.append({
                'timestep': timestep,
                'update': update + 1,
                'task': current_task,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'best_reward': best_reward,
                'entropy': float(metrics['entropy']),
                'pg_loss': float(metrics['pg_loss']),
                'vf_loss': float(metrics['vf_loss']),
                'ent_coef': float(ent_coef) if method in ['trac'] else args.ent_coef,
                'noise_magnitude': float(jnp.linalg.norm(noise_vector)) if task_type == 'noise' else 0.0,
                'param_value': float(getattr(env_params, param_name)) if task_type == 'param' else None,
                'elapsed_time': elapsed,
            })
            
            log_dict = {
                'timestep': timestep,
                'task': current_task,
                'eval/mean_reward': mean_reward,
                'eval/best_reward': best_reward,
                'train/entropy': metrics['entropy'],
                'train/pg_loss': metrics['pg_loss'],
                'train/vf_loss': metrics['vf_loss'],
                'train/ent_coef': ent_coef if method == 'trac' else args.ent_coef,
            }
            if task_type == 'noise':
                log_dict['noise_magnitude'] = float(jnp.linalg.norm(noise_vector))
            elif task_type == 'param':
                log_dict[param_name] = float(getattr(env_params, param_name))
            wandb.log(log_dict)
            
            print(f"Update {update+1:5d} | Task {current_task} | Timestep {timestep:10,} | "
                  f"Reward: {mean_reward:8.2f} ± {std_reward:.2f} | Best: {best_reward:8.2f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s, Best (training): {best_reward:.2f}")
    
    # Final evaluation with 10 trials
    if task_type == 'noise':
        print(f"\nFinal evaluation (10 trials) on Task {current_task} with noise magnitude {float(jnp.linalg.norm(noise_vector)):.4f}...")
    elif task_type == 'param':
        print(f"\nFinal evaluation (10 trials) on Task {current_task} with {param_name} {float(getattr(env_params, param_name)):.4f}...")
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
        final_task_name = f"{param_name}_{float(getattr(env_params, param_name)):.4f}"
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
        final_ckpt_data['param_value'] = float(getattr(env_params, param_name))
    with open(task_ckpt_path, 'wb') as f:
        pickle.dump(final_ckpt_data, f)
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
    
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
