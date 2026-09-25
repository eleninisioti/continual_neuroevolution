"""
Train PPO on Gymnax environments (non-continual).

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
- ReDo: Recycling Dormant Neurons (Sokar et al., ICML 2023); see source/algorithms/rl/redo.py
- C-CHAIN: Churn Approximated Reduction (Tang et al., 2025); the churn
  regularizer applies within a single MDP too, so it is available here as well
  (see source/studies/gymnax/cchain.py)

Usage:
    python train_RL_gymnax.py --env CartPole-v1 --method ppo --gpus 0
    python train_RL_gymnax.py --env MountainCar-v0 --method cchain --gpus 0
    python train_RL_gymnax.py --env Acrobot-v1 --method trac --gpus 0
    python train_RL_gymnax.py --env MountainCar-v0 --method redo --gpus 0
"""

import argparse
import functools
import os
import sys
import time
import pickle
import json

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
from source.utils.runtime import Tee, _get_gpu_arg, write_run_config

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
from source.envs.gymnax_classic import make_gymnax_env, wrap_actions
import wandb
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')  # Headless backend for GIF rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.lines import Line2D

from source.algorithms.rl import redo
from source.metrics import plasticity
from source.metrics import ntk as ntk_metrics
from source.metrics.plasticity import policy_churn_cross_entropy as chain_policy_churn
from source.metrics.weight_stats import weight_stats
from source.algorithms.networks import ACTIVATIONS
from source.algorithms.rl.ppo import GYMNAX_POLICY_ACTIVATION, GYMNAX_VALUE_ACTIVATION
from source.metrics.behaviour_descriptors import collect_probe_states

# source/studies/gymnax/cchain.py is not present in every checkout (it is only needed by
# --method cchain). Keep the other methods importable when it is missing.
try:
    from source.studies.gymnax.cchain import (
        ChainCoefController, add_chain_args, init_chain_state, make_chain_sgd_epochs,
    )
    _CCHAIN_AVAILABLE = True
except ImportError:
    _CCHAIN_AVAILABLE = False

    def add_chain_args(parser):
        """No-op stand-in so --help and the other methods still work."""
        return parser

    def _cchain_missing(*args, **kwargs):
        raise ImportError(
            "--method cchain requires source/studies/gymnax/cchain.py, which is missing "
            "from this checkout."
        )

    ChainCoefController = _cchain_missing
    init_chain_state = _cchain_missing
    make_chain_sgd_epochs = _cchain_missing


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
    env_carry=None,
):
    """Collect rollout data from parallel environments using jax.lax.scan (GPU-friendly).

    `env_carry` is the (obs, states) the previous rollout stopped on. Passing it
    is what makes a rollout a sliding window over episodes that are still
    running, rather than num_steps steps from a fresh reset.

    That distinction is the whole ballgame on Acrobot and MountainCar. Resetting
    here meant every update trained on steps 0..num_steps-1 of a new episode,
    and neither task can reach its goal that fast: measured over 512 episodes of
    a random policy, 0 of them terminate within 50 steps and the median
    termination is the 500-step time limit. So `dones` was all-False and
    `rewards` a constant -1 in every batch -- no reward variance, hence no
    policy-gradient signal, hence the entropy collapse to ~0.01 by 10% of the
    run. CartPole was unharmed only because a good CartPole policy survives the
    window, so its data was never truncated away.

    Pass None for the first rollout of a run, which resets.

    Returns (rollout, next_carry, last_obs). `last_obs` is the state the window
    stopped on; GAE bootstraps V(last_obs) instead of 0.
    """

    if env_carry is None:
        key, reset_key = random.split(key)
        reset_keys = random.split(reset_key, num_envs)
        obs, states = jax.vmap(lambda k: env.reset(k, env_params))(reset_keys)
    else:
        obs, states = env_carry

    def env_step(carry, _):
        obs, states, key = carry
        
        # Get policy output
        logits = policy_network.apply(policy_params, obs)
        values = value_network.apply(value_params, obs)
        
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
        
        # Store transition data
        transition = {
            'obs': obs,
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

    # rollout is a dict with arrays of shape (num_steps, num_envs, ...)
    return rollout, (final_obs, final_states), final_obs




# ============================================================================
# ReDo lives in source/algorithms/rl/redo.py, shared with train_RL_gymnax_continual.py.
# ============================================================================



# ============================================================================
# PT: Permanent-Transient Value Decomposition
# ============================================================================

def compute_pt_ppo_loss(
    policy_params,
    value_params_t,
    value_params_p,
    policy_network,
    value_network,
    batch,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
):
    """PPO loss with PT value decomposition. Only V_T receives gradients."""
    obs = batch['obs']
    actions = batch['actions']
    old_log_probs = batch['log_probs']
    advantages = batch['advantages']
    returns = batch['returns']

    logits = policy_network.apply(policy_params, obs)
    values_t = value_network.apply(value_params_t, obs)
    values_p = jax.lax.stop_gradient(value_network.apply(value_params_p, obs))
    values = values_p + values_t

    log_probs = jax.vmap(categorical_log_prob)(logits, actions)
    ratio = jnp.exp(log_probs - old_log_probs)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
    vf_loss = 0.5 * jnp.square(values - returns).mean()
    entropy = jax.vmap(categorical_entropy)(logits).mean()
    loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy

    return loss, {
        'pg_loss': pg_loss,
        'vf_loss': vf_loss,
        'entropy': entropy,
        'approx_kl': jnp.mean((ratio - 1) - jnp.log(ratio)),
    }


def pt_train_step(
    policy_network,
    value_network,
    clip_eps,
    vf_coef,
    policy_state,
    value_state_t,
    value_params_p,
    batch,
    ent_coef,
):
    """PPO training step with PT value decomposition. Updates policy and V_T only."""
    def loss_fn(policy_params, value_params_t):
        return compute_pt_ppo_loss(
            policy_params, value_params_t, value_params_p,
            policy_network, value_network,
            batch, clip_eps, vf_coef, ent_coef,
        )

    (loss, metrics), (policy_grads, value_t_grads) = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(policy_state.params, value_state_t.params)

    policy_state = policy_state.apply_gradients(grads=policy_grads)
    value_state_t = value_state_t.apply_gradients(grads=value_t_grads)

    return policy_state, value_state_t, loss, metrics


def pt_decay_transient(value_state_t, decay):
    """Decay transient value network weights after absorption."""
    new_params = jax.tree_util.tree_map(lambda p: p * decay, value_state_t.params)
    return value_state_t.replace(params=new_params)


# ============================================================================
# TRAC: Automatic Entropy Coefficient Tuning
# ============================================================================


# ============================================================================
# Evaluation
# ============================================================================

def make_eval_fn(env, env_params, policy_network, max_steps, num_episodes):
    """Build a jitted greedy-rollout evaluator.

    Episodes run for a fixed `max_steps` under lax.scan and are vmapped over,
    so a whole evaluation is one device call with no host sync. Rewards after
    termination are masked out rather than breaking the loop, which is what
    makes the fixed-length scan equivalent to stopping at `done`. This mirrors
    the scoring function in the NE trainers.
    """

    def rollout(policy_params, key):
        reset_key, scan_key = random.split(key)
        obs, state = env.reset(reset_key, env_params)

        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
            logits = policy_network.apply(policy_params, obs)
            action = jnp.argmax(logits)  # Greedy action for eval

            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(
                step_key, state, action, env_params
            )
            total_reward = total_reward + reward * (1.0 - done_flag)
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))

            return (next_obs, next_state, total_reward, done_flag, key), None

        init = (obs, state, 0.0, 0.0, scan_key)
        (_, _, total_reward, _, _), _ = jax.lax.scan(
            step_fn, init, None, length=max_steps
        )
        return total_reward

    @jax.jit
    def eval_fn(policy_params, key):
        keys = random.split(key, num_episodes)
        rewards = jax.vmap(rollout, in_axes=(None, 0))(policy_params, keys)
        return rewards

    return eval_fn


def make_gif_rollout_fn(env, env_params, policy_network, max_steps):
    """Jitted rollout that also returns the observation trajectory for rendering.

    Rendering itself stays in Python (matplotlib), but the environment stepping
    does not: the whole trajectory comes back from one device call, along with
    the number of steps before termination so the caller can trim the frames.
    """

    @jax.jit
    def rollout(policy_params, key):
        reset_key, scan_key = random.split(key)
        obs, state = env.reset(reset_key, env_params)

        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
            logits = policy_network.apply(policy_params, obs)
            action = jnp.argmax(logits)

            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(
                step_key, state, action, env_params
            )
            total_reward = total_reward + reward * (1.0 - done_flag)
            alive = 1.0 - done_flag
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))

            return (next_obs, next_state, total_reward, done_flag, key), (obs, alive)

        init = (obs, state, 0.0, 0.0, scan_key)
        (_, _, total_reward, _, _), (obs_traj, alive) = jax.lax.scan(
            step_fn, init, None, length=max_steps
        )
        # alive[t] is 1.0 while the episode is still running at step t.
        num_steps = jnp.sum(alive).astype(jnp.int32)
        return obs_traj, total_reward, num_steps

    return rollout


# ============================================================================
# Environment-Specific Hyperparameters
#
# THE GYMNAX RL ARM DOES NOT NORMALISE OBSERVATIONS. This is deliberate, and
# every one of these dicts used to carry a `normalize_observations: True` that
# get_hyperparams() never read -- so the config asserted normalisation that no
# gymnax run has ever done, on any env, in any of the four gymnax trainers. The
# key was deleted on 2026-08-06 rather than implemented; this note is what
# replaces it, so the absence stays a recorded choice instead of looking like an
# oversight to whoever reads this next.
#
# Why not normalise: it is not the convention for discrete classic control.
# purejaxrl's gymnax PPO and CleanRL's ppo.py both skip it, and those are the
# implementations these numbers get compared against. The observations are
# low-dimensional and roughly O(1): measured per-dimension std over 256 episodes
# x 500 steps is 8.6x spread on CartPole, 11.5x on Acrobot, 12x on MountainCar
# -- not the 1000x that makes normalisation mandatory in continuous control.
#
# Note the asymmetry with the rest of the study, which is a domain convention
# and not an inconsistency: source/studies/brax/my_brax DOES normalise (brax's
# running_statistics, --normalize_observations default True), because MuJoCo and
# Brax observations mix joint angles, velocities and contact forces across
# orders of magnitude. Kinetix does not normalise either.
#
# The case for revisiting it, if anyone does: MountainCar's velocity has std
# 0.010 against position's 0.120, and velocity is the dimension the task is
# actually about. Acrobot has the mirror problem, its angular velocities
# dominating the cos/sin terms ~11x. That is a real experiment (3 seeds x 2 envs
# at full budget, normalised against not), not a cleanup -- and it would re-run
# the whole RL arm, since all four methods share these dicts.
# ============================================================================

ENV_CONFIGS = {
    "CartPole-v1": {
        "num_timesteps": 512 * 500 * 50,  # 204,800,000
        "num_envs": 2048,
        # 50, not the 20 this used to carry. Every other gymnax PPO config --
        # Acrobot and MountainCar here, and all three in
        # train_RL_gymnax_continual.py -- unrolls 50, and the continual block
        # derives its switch interval from 2048 x 50 = 102,400 steps per
        # update. At 20 the stationary CartPole reference ran the same env-step
        # budget as 11,250 updates against the continual run's 15,000, i.e. a
        # different batch size and GAE horizon. Forward transfer subtracts the
        # two, so that landed in the FT column: FT for sub-task 0 -- the
        # no-noise baseline, where it must be 0 -- came out at +0.215 (PPO),
        # +0.183 (TRAC), +0.334 (ReDo) and +0.093 (C-CHAIN) on CartPole, while
        # Acrobot and MountainCar, whose configs already matched, were within
        # 0.02 of 0.
        "num_steps": 50,  # unroll_length
        "num_epochs": 10,  # num_updates_per_batch
        "num_minibatches": 32,
        "gamma": 0.95,  # discounting
        "learning_rate": 3e-4,
        "ent_coef": 1e-2,
        "policy_hidden_dims": (16, 16),
        "value_hidden_dims": (128, 128, 128),
        "episode_length": 500,
    },
    "Acrobot-v1": {
        "num_timesteps": 512 * 500 * 200 * 10,  # 512,000,000
        "num_envs": 2048,
        "num_steps": 50,  # unroll_length
        "num_epochs": 10,  # num_updates_per_batch
        "num_minibatches": 32,
        "gamma": 0.99,  # discounting
        "learning_rate": 1e-4,
        "ent_coef": 1e-2,
        "policy_hidden_dims": (16, 16),
        "value_hidden_dims": (128, 128, 128),
        "episode_length": 500,
    },
    "MountainCar-v0": {
        "num_timesteps": 512 * 500 * 200 * 20,  # 1,024,000,000
        "num_envs": 2048,
        "num_steps": 50,  # unroll_length
        "num_epochs": 10,  # num_updates_per_batch
        # 0.99, not the 0.95 this carried until 2026-08-06. At 0.95 over a
        # 500-step episode the effective horizon is ~20 steps, and MountainCar
        # needs ~100 to reach the goal -- so a state one push away from success
        # discounts to nearly the same return as one that never gets there.
        # Measured at the full compute-matched budget, 3 seeds, after the
        # rollout fix:
        #
        #            final            peak    peak-final
        #   0.95   -114.4 +- 11.8    -93.9        20.4
        #   0.99    -97.2            -89.1         8.1
        #
        # Both clear the -130 target, so this is not about reaching it: it is
        # the `peak - final` column. Both discounts find a good policy and only
        # 0.99 holds it, which is the same late-run degradation the rollout bug
        # produced, from a different cause. Acrobot was already 0.99; CartPole
        # stays at 0.95, where its 500-step episodes are the task rather than a
        # journey to a sparse goal.
        "num_minibatches": 32,
        "gamma": 0.99,  # discounting
        "learning_rate": 3e-4,
        "ent_coef": 1e-2,
        "policy_hidden_dims": (16, 16),
        "value_hidden_dims": (128, 128, 128),
        "episode_length": 500,
    },
}


# ============================================================================
# Main Training Loop
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='PPO on Gymnax (Non-Continual)')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'])
    parser.add_argument('--method', type=str, default='ppo',
                        choices=['ppo', 'trac', 'redo', 'pt', 'cchain'])
    parser.add_argument('--num_timesteps', type=int, default=None,
                        help='Override default timesteps for env')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)
    
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
    parser.add_argument('--anneal_lr', type=int, default=0,
                        help='Linearly decay the learning rate to 0 over the run.')
    parser.add_argument('--max_grad_norm', type=float, default=0.0,
                        help='Global-norm gradient clipping; <=0 disables it.')
    
    # TRAC (Muppidi et al., NeurIPS 2024) takes no hyperparameters -- that is the
    # point of it. `--method trac` wraps the optimizer with `start_trac` and the
    # tuner's own constants (eps, s_prev, num_betas) are left at the package
    # defaults, which is also what brax, mujoco and kinetix pass.

    # ReDo specific (Sokar et al. 2023; see source/algorithms/rl/redo.py)
    parser.add_argument('--redo_interval', type=int, default=50,
                        help='Apply ReDo every N updates')
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
    parser.add_argument('--num_probe_states', type=int, default=512,
                        help='Size of the frozen probe batch used for the churn and '
                             'dormancy diagnostics. Same name and default as the NE '
                             'trainers use, so both sides measure on batches of equal size.')
    parser.add_argument('--track_dormant', action='store_true',
                        help='Log dormant-neuron counts at every eval for any method, so a '
                             'baseline can be compared against ReDo. Implied by --method redo.')

    # PT (Permanent-Transient) specific
    parser.add_argument('--pt_update_interval', type=int, default=50,
                        help='Absorb T into P every N updates')
    parser.add_argument('--pt_decay', type=float, default=0.0,
                        help='Decay factor for T weights after absorption (0=full reset)')
    parser.add_argument('--pt_perm_lr', type=float, default=0.01,
                        help='Learning rate for permanent value network (SGD)')
    parser.add_argument('--pt_absorb_steps', type=int, default=10,
                        help='Number of SGD steps for P to absorb T')
    
    # C-CHAIN specific
    add_chain_args(parser)

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
    seed = args.seed
    trial = args.trial
    
    # Get environment-specific hyperparameters
    hp = get_hyperparams(args)
    num_timesteps = hp['num_timesteps']
    
    print("=" * 60)
    print(f"PPO ({method.upper()}) on {env_name} (Non-Continual)")
    print("=" * 60)
    
    # Create environment
    env, env_params = make_gymnax_env(env_name)
    env_params = env_params.replace(max_steps_in_episode=hp['episode_length'])
    obs_dim = env.observation_space(env_params).shape[0]
    action_dim = env.action_space(env_params).n
    
    print(f"  Environment: {env_name}")
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"  Method: {method.upper()}")
    print(f"  Total timesteps: {num_timesteps:,}")
    print(f"  Hyperparams: num_envs={hp['num_envs']}, num_steps={hp['num_steps']}, "
          f"lr={hp['learning_rate']}, gamma={hp['gamma']}, ent_coef={hp['ent_coef']}")
    
    # Output directory
    if args.output_dir is None:
        output_dir = os.path.join(
            REPO_ROOT, "projects", "gymnax",
            f"{method}_{env_name.replace('-', '_')}",
            f"trial_{trial}"
        )
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, "train.log")
    sys.stdout = Tee(log_file)
    
    print(f"  Output: {output_dir}")
    
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
    # Two knobs the study's PPO did not have, both standard and both aimed at
    # the same failure -- a policy that reaches a good region and then leaves it
    # (Acrobot peaks at -71 and ends at -86; MountainCar peaks at -126 and ends
    # at -220 +- 140). A constant learning rate keeps taking full-size steps
    # after the batch stops carrying new information, and an unclipped gradient
    # lets one bad batch move the policy arbitrarily far.
    #
    # max_grad_norm <= 0 disables clipping; anneal_lr 0 keeps the rate constant,
    # which together reproduce the previous optimiser exactly.
    _updates_total = max(1, hp['num_timesteps'] // (hp['num_envs'] * hp['num_steps']))
    if args.anneal_lr:
        lr_schedule = optax.linear_schedule(
            init_value=hp['learning_rate'],
            end_value=0.0,
            # One SGD step per minibatch per epoch, which is what the optimiser
            # actually counts -- not one per update.
            transition_steps=_updates_total * hp['num_epochs'] * hp['num_minibatches'],
        )
    else:
        lr_schedule = hp['learning_rate']

    def make_base_optimizer():
        if args.max_grad_norm > 0:
            return optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(lr_schedule),
            )
        return optax.adam(lr_schedule)

    # TRAC is an optimiser wrapper, not a loss or an exploration term: it
    # rescales the whole update by a parameter-free tuner. Applied outside the
    # clipping so it sees the gradient after clipping, matching brax's and
    # kinetix's wiring (`my_brax/ppo_train.py`, `kinetix/experiments/ppo.py`).
    #
    # ONE tuner over the policy AND the value parameters, not one per network.
    # This is the only suite where that distinction exists: brax, mujoco and
    # kinetix each have a single actor-critic network, so their single
    # `start_trac` is joint by construction, and the C-CHAIN reference for this
    # exact setting -- separate policy and value nets on classic control --
    # builds one TRAC over both param groups
    # (`inspiration/C-CHAIN/crl_gym_classic_control/control_train_aligned_vanilla.py`).
    #
    # It is not cosmetic. The tuner is six scalars over whatever pytree it is
    # handed, driven by h = <grad, theta_ref - theta>. Handed the 435-parameter
    # policy alone, h after convergence is dominated by the entropy bonus --
    # PPO's clip has zeroed the surrogate for most samples -- which pulls back
    # toward the near-uniform initialisation. The tuner reads that as "the
    # displacement is not paying off", drives its scale to zero, and writes
    # theta = theta_ref: the policy is reset to its initial weights mid-run.
    # That is the -500 cliff in runs_repro2 (5/10 Acrobot and 7/10 MountainCar
    # trials; the collapsed steps sit at exactly the initial weight norm and at
    # entropy log 3). Joint with the 34,049-parameter critic, whose gradient
    # stays informative for the whole run, the scale has no such excursion.
    #
    # Measured on Acrobot over 200 updates, logging the tuner state of each of
    # the two split wrappers. sum(s) is the scale multiplying the displacement:
    #
    #     policy-only   5.9e-08 -> 8.2 -> 19.7 -> 1.7 -> 11.4 -> 26.5 -> 35.3
    #     value-only    1.9 -> 2.8 -> 3.4 -> 3.6 -> 3.1 -> 3.8 -> 4.5
    #
    # The policy's scale spans 20x and crashes by 10x at update 80 before
    # recovering; the critic's stays inside a factor of 3. A collapse to
    # sum(s) <= 0 is the tail of that same volatility, which is why it hits
    # some seeds and not others, at updates as far apart as 160 and 840. The
    # tuner's accumulated evidence sum(sigma) is ~3e4 on the critic against
    # ~5e1 on the policy, so in a joint tuner -- where h is summed over both
    # pytrees -- the critic dominates the shared scalar by three orders of
    # magnitude and the policy's noise cannot drag it to zero.
    #
    # Not the fix, and checked: the pip JAX port drops the `max(sum(s), 0)`
    # clamp that `trac_optimizer/trac.py` applies at the parameter write.
    # Restoring it is strictly worse -- sum(s)=0 gives theta=theta_ref, hence
    # delta_prev=0, hence h=0 forever, an absorbing state. Acrobot trial 10
    # with the clamp sat at -500 from update 10 to 200 without ever learning.
    joint_trac_tx = None
    if method == 'trac':
        joint_trac_tx = start_trac(make_base_optimizer())

    policy_optimizer = make_base_optimizer()
    value_optimizer = make_base_optimizer()

    print(f"  Optimizer: adam lr={hp['learning_rate']:g}"
          f"{' (annealed to 0)' if args.anneal_lr else ''}"
          f", grad clip={args.max_grad_norm if args.max_grad_norm > 0 else 'off'}"
          f"{', TRAC wrapper (joint policy+value)' if method == 'trac' else ''}")


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
    # The single TRAC state, threaded through the epoch loop by hand because it
    # spans two TrainStates. None for every other method.
    joint_opt_state = None
    if joint_trac_tx is not None:
        joint_opt_state = joint_trac_tx.init(
            {'policy': policy_state.params, 'value': value_state.params})
    
    # PT: Initialize permanent value network
    value_state_p = None
    if method == 'pt':
        key, perm_key = random.split(key)
        perm_value_params = value_network.init(perm_key, dummy_obs)
        perm_optimizer = optax.sgd(args.pt_perm_lr)
        value_state_p = TrainState.create(
            apply_fn=value_network.apply,
            params=perm_value_params,
            tx=perm_optimizer,
        )
        perm_param_count = sum(p.size for p in jax.tree_util.tree_leaves(perm_value_params))
        print(f"  PT: V_P params: {perm_param_count:,}, V_T params: {value_param_count:,}")
        print(f"  PT: update_interval={args.pt_update_interval}, decay={args.pt_decay}, "
              f"perm_lr={args.pt_perm_lr}, absorb_steps={args.pt_absorb_steps}")
    
    # C-CHAIN: Initialize reference networks and coefficient controller
    chain_state = None
    chain_ctrl = None
    jit_sgd_epochs_chain = None
    if method == 'cchain':
        chain_state = init_chain_state(policy_state, value_state)
        chain_ctrl = ChainCoefController(
            args.chain_target_rel_scale, args.chain_warmup_updates, args.chain_coef_window
        )
        jit_sgd_epochs_chain = make_chain_sgd_epochs(
            policy_network, value_network, compute_ppo_loss,
            args.clip_eps, args.vf_coef, hp['num_epochs'], hp['num_minibatches'],
            hp['num_steps'] * hp['num_envs'],
        )
        print(f"  C-CHAIN: target_rel_scale={args.chain_target_rel_scale}, "
              f"warmup_updates={args.chain_warmup_updates}, "
              f"coef_window={args.chain_coef_window}")

    # Initialize wandb
    config = {
        'env': env_name, 'method': method, 'num_timesteps': num_timesteps,
        'seed': seed, 'trial': trial, 'num_envs': hp['num_envs'],
        'num_steps': hp['num_steps'], 'learning_rate': hp['learning_rate'],
        'gamma': hp['gamma'], 'ent_coef': hp['ent_coef'],
        'num_epochs': hp['num_epochs'], 'num_minibatches': hp['num_minibatches'],
        'policy_hidden_dims': hp['policy_hidden_dims'], 'value_hidden_dims': hp['value_hidden_dims'],
    }
    if method == 'cchain':
        config.update({
            'chain_target_rel_scale': args.chain_target_rel_scale,
            'chain_warmup_updates': args.chain_warmup_updates,
            'chain_coef_window': args.chain_coef_window,
        })

    # ReDo always tracks dormancy; other methods only if asked, so the baseline
    # curves can be plotted next to it.
    track_dormant = args.track_dormant or method == 'redo'
    if method == 'redo':
        config.update({
            'redo_interval': args.redo_interval,
            'redo_tau': args.redo_tau,
            'redo_targets': args.redo_targets,
            'redo_batch_size': args.redo_batch_size,
            'redo_reset_adam_count': not args.redo_keep_adam_count,
        })
        print(f"  ReDo: every {args.redo_interval} updates, tau={args.redo_tau}, "
              f"targets={args.redo_targets}, "
              f"reset_adam_count={not args.redo_keep_adam_count}")
    wandb.init(project=args.wandb_project, config=config,
               name=f"{method}_{env_name}_trial{trial}", reinit=True)
    # The run says what network it searched, so
    # scripts/check_architectures.py can confirm every method compared on
    # this task used the same one. The continual trainers already wrote
    # this into results.json; the stationary ones wrote it nowhere.
    write_run_config(output_dir, config, policy_arch='gymnax')
    
    # `ent_coef` is a plain constant for every method here. It used to be state
    # that `--method trac` mutated; see the note on start_trac above.
    ent_coef = hp['ent_coef']

    # Training metrics
    best_reward = -float('inf')
    training_metrics = []
    start_time = time.time()

    # Jitted greedy evaluator, reused for the periodic evals and the final one.
    eval_fn = make_eval_fn(
        env, env_params, policy_network,
        hp['episode_length'], args.num_eval_episodes,
    )

    # Calculate number of updates
    timesteps_per_update = hp['num_envs'] * hp['num_steps']
    num_updates = num_timesteps // timesteps_per_update
    
    print(f"\nStarting training ({num_updates} updates)...")
    
    # JIT compile training step (static args first via partial)
    jit_train_step = jax.jit(functools.partial(
        train_step,
        policy_network,  # Fixed arg 1
        value_network,   # Fixed arg 2
        hp['clip_eps'],  # Fixed arg 3
        hp['vf_coef'],   # Fixed arg 4
    ))

    # TRAC: the same step, but driving one optimiser over both parameter sets.
    jit_train_step_joint = None
    if joint_trac_tx is not None:
        jit_train_step_joint = jax.jit(functools.partial(
            train_step_joint,
            policy_network,
            value_network,
            hp['clip_eps'],
            hp['vf_coef'],
            joint_trac_tx,
        ))

    # PT: JIT compile PT-specific functions
    jit_pt_train_step = jax.jit(functools.partial(
        pt_train_step,
        policy_network,
        value_network,
        hp['clip_eps'],
        hp['vf_coef'],
    ))
    
    @jax.jit
    def jit_add_perm_values(rollout_obs, rollout_values, perm_params):
        """Add permanent value network predictions to rollout values."""
        S, N = rollout_values.shape
        D = rollout_obs.shape[-1]
        v_p = value_network.apply(perm_params, rollout_obs.reshape(S * N, D)).reshape(S, N)
        return rollout_values + v_p
    
    pt_absorb_steps = args.pt_absorb_steps
    
    @jax.jit
    def jit_pt_absorb(value_state_p, value_params_t, obs_batch):
        """Train V_P to absorb V_T's knowledge."""
        targets = jax.lax.stop_gradient(
            value_network.apply(value_state_p.params, obs_batch)
            + value_network.apply(value_params_t, obs_batch)
        )
        def p_step(state, _):
            def p_loss(p_params):
                pred = value_network.apply(p_params, obs_batch)
                return 0.5 * jnp.square(pred - targets).mean()
            loss, grads = jax.value_and_grad(p_loss)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss
        value_state_p, losses = jax.lax.scan(
            p_step, value_state_p, None, length=pt_absorb_steps
        )
        return value_state_p, losses[-1]
    
    # JIT compile rollout collection (fix static args)
    @jax.jit
    def jit_collect_rollout_fn(key, policy_params, value_params, env_carry):
        return collect_rollout(
            key, policy_network, policy_params,
            value_network, value_params,
            env, env_params, hp['num_envs'], hp['num_steps'],
            env_carry=env_carry,
        )

    # The first rollout of the run has no previous window to continue from.
    @jax.jit
    def jit_init_env_carry(key):
        reset_keys = random.split(key, hp['num_envs'])
        return jax.vmap(lambda k: env.reset(k, env_params))(reset_keys)

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
    
    key, carry_key = random.split(key)
    env_carry = jit_init_env_carry(carry_key)

    # --- Plasticity diagnostics (source/metrics/plasticity.py) ------------------
    # Reported for EVERY method, so dormancy and churn are columns of the same
    # table for ppo, trac, redo, cchain and the NE methods alike.
    #
    # Both are pure observers: the probe batch is drawn from a private RNG
    # stream (the `seed + N_000_000` convention the diversity tracking already
    # uses), and nothing here writes back into training. A run with these on and
    # one with them off are the same run, bit for bit.
    probe_key = random.key(seed + 3_000_000)
    probe_obs = None                      # frozen after the first rollout
    prev_policy_params = None             # for the churn measurement
    _policy_act_fn = ACTIVATIONS[GYMNAX_POLICY_ACTIVATION]
    _value_act_fn = ACTIVATIONS[GYMNAX_VALUE_ACTIVATION]
    _policy_criterion = redo.criterion_for_activation(GYMNAX_POLICY_ACTIVATION)
    _value_criterion = redo.criterion_for_activation(GYMNAX_VALUE_ACTIVATION)

    def _policy_logits(params, obs_batch):
        return policy_network.apply(params, obs_batch)

    for update in range(num_updates):
        timestep = (update + 1) * timesteps_per_update

        # Collect rollout. env_carry threads the environments across updates, so
        # this window picks up where the last one stopped instead of resetting.
        key, rollout_key = random.split(key)
        rollout, env_carry, last_obs = jit_collect_rollout_fn(
            rollout_key, policy_state.params, value_state.params, env_carry
        )

        last_values = value_network.apply(value_state.params, last_obs)

        # Freeze the churn/dormancy probe states once, from the first rollout.
        # Fixed states are the point: measuring on each update's own rollout
        # would confound "the policy changed" with "the state distribution
        # moved". See source/metrics/plasticity.py.
        if probe_obs is None:
            probe_obs = collect_probe_states(
                rollout['obs'], num_probe=args.num_probe_states, key=probe_key)

        # The policy as of before this update, for the churn measurement below.
        prev_policy_params = policy_state.params

        # PT: add permanent value predictions to rollout values
        if method == 'pt':
            combined_values = jit_add_perm_values(
                rollout['obs'], rollout['values'], value_state_p.params
            )
            rollout = {**rollout, 'values': combined_values}
            # The bootstrap has to be on the same scale as the values GAE sees.
            last_values = last_values + value_network.apply(
                value_state_p.params, last_obs
            )

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
        
        # Training epochs
        if method == 'cchain':
            # The whole epoch/minibatch loop runs inside one jitted scan
            key, sgd_key = random.split(key)
            policy_state, value_state, chain_state, loss, metrics = jit_sgd_epochs_chain(
                policy_state, value_state, flat_batch, ent_coef,
                chain_ctrl.coef, chain_state, sgd_key
            )
            chain_ctrl.update(metrics['chain_p_loss'], metrics['chain_p_reg_loss'], update)
        else:
            for epoch in range(hp['num_epochs']):
                # Shuffle
                key, shuffle_key = random.split(key)
                perm = random.permutation(shuffle_key, batch_size)
                shuffled_batch = {k: v[perm] for k, v in flat_batch.items()}
            
                # Minibatches
                minibatch_size = batch_size // hp['num_minibatches']
                for mb in range(hp['num_minibatches']):
                    start_idx = mb * minibatch_size
                    end_idx = start_idx + minibatch_size
                    minibatch = {k: v[start_idx:end_idx] for k, v in shuffled_batch.items()}
                
                    if method == 'pt':
                        policy_state, value_state, loss, metrics = jit_pt_train_step(
                            policy_state, value_state, value_state_p.params,
                            minibatch, ent_coef
                        )
                    elif jit_train_step_joint is not None:
                        policy_state, value_state, joint_opt_state, loss, metrics = (
                            jit_train_step_joint(
                                policy_state, value_state, joint_opt_state,
                                minibatch, ent_coef
                            ))
                    else:
                        policy_state, value_state, loss, metrics = jit_train_step(
                            policy_state, value_state, minibatch, ent_coef
                        )

        # TRAC needs no per-update hook: the tuner lives inside the optimizer
        # and has already run, once per gradient step, inside the loop above.

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
            if (update + 1) % args.eval_interval == 0:
                summary = ' | '.join(
                    f"{n}: {s['dormant_count']}/{s['total_neurons']} "
                    f"({s['dormant_fraction']:.1%})" for n, s in redo_stats.items()
                )
                print(f"  ReDo recycled -> {summary}")

        # PT: Absorb transient into permanent, then decay transient
        if method == 'pt' and (update + 1) % args.pt_update_interval == 0:
            obs_sample = flat_batch['obs'][:min(2048, flat_batch['obs'].shape[0])]
            value_state_p, absorb_loss = jit_pt_absorb(
                value_state_p, value_state.params, obs_sample
            )
            value_state = pt_decay_transient(value_state, args.pt_decay)
            if (update + 1) % args.eval_interval == 0:
                print(f"  PT absorb: loss={float(absorb_loss):.6f}")
        
        # Evaluate
        if (update + 1) % args.eval_interval == 0 or update == num_updates - 1:
            key, eval_key = random.split(key)
            eval_rewards = np.asarray(eval_fn(policy_state.params, eval_key))
            mean_reward, std_reward = float(eval_rewards.mean()), float(eval_rewards.std())

            if mean_reward > best_reward:
                best_reward = mean_reward

            elapsed = time.time() - start_time

            # Dormant-neuron diagnostic, measured *after* any ReDo pass. Scored
            # on the frozen probe batch, and with the criterion ReDo itself
            # recycles on, so this column means the same thing for every method
            # and is comparable with the NE trainers' population dormancy.
            #
            # Unconditional. It used to be gated on `--track_dormant or method
            # == 'redo'`, and the gymnax block never passed the flag, so only
            # the `redo` arm ever recorded it -- ppo, trac and cchain wrote
            # null. It is a pure measurement (no RNG, no state), so switching it
            # on cannot move a trained number.
            dormant = {}
            dormant['policy'] = redo.dormant_stats(
                policy_state.params, probe_obs, len(hp['policy_hidden_dims']),
                args.redo_tau, activation_fn=_policy_act_fn,
                criterion=_policy_criterion)
            dormant['value'] = redo.dormant_stats(
                value_state.params, probe_obs, len(hp['value_hidden_dims']),
                args.redo_tau, activation_fn=_value_act_fn,
                criterion=_value_criterion)

            # Churn: fraction of probe states whose greedy action changed over
            # this update. Same definition the NE trainers use between
            # generations, so the columns are comparable. NOT the same quantity
            # as C-CHAIN's cross-entropy `chain/policy_churn`, which is the
            # regulariser's own per-gradient-step control signal.
            churn = float(plasticity.churn_action_disagreement(
                _policy_logits, prev_policy_params, policy_state.params, probe_obs))

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

            # Weight statistics, the third plasticity signal alongside dormancy
            # and churn -- the failure those two miss is a norm that grows
            # without bound until the network is hard to move while every unit
            # is still nominally active. Policy and value separately: they have
            # different parameter counts, so only the `_rms` columns are
            # comparable between them.
            _wstats = {
                **weight_stats(policy_state.params, prefix='policy_weight'),
                **weight_stats(value_state.params, prefix='value_weight'),
            }

            training_metrics.append({
                'timestep': timestep,
                'update': update + 1,
                'policy_churn_action': churn,
                **_wstats,
                'policy_dormant_fraction_probe': dormant['policy']['dormant_fraction'],
                'value_dormant_fraction_probe': dormant['value']['dormant_fraction'],
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'best_reward': best_reward,
                'entropy': float(metrics['entropy']),
                'pg_loss': float(metrics['pg_loss']),
                'vf_loss': float(metrics['vf_loss']),
                'ent_coef': args.ent_coef,
                'elapsed_time': elapsed,
                'chain_p_reg_coef': chain_ctrl.coef if method == 'cchain' else None,
                'chain_p_reg_loss': float(metrics['chain_p_reg_loss']) if method == 'cchain' else None,
                'policy_churn': _churn_ce,
                **_ntk,
                'value_churn': float(metrics['value_churn']) if method == 'cchain' else None,
                'policy_dormant_fraction': dormant['policy']['dormant_fraction'] if dormant else None,
                'value_dormant_fraction': dormant['value']['dormant_fraction'] if dormant else None,
            })

            log_dict = {
                'timestep': timestep,
                'plasticity/policy_churn_action': churn,
                'plasticity/policy_dormant_fraction': dormant['policy']['dormant_fraction'],
                'plasticity/value_dormant_fraction': dormant['value']['dormant_fraction'],
                'eval/mean_reward': mean_reward,
                'eval/best_reward': best_reward,
                'train/entropy': metrics['entropy'],
                'train/pg_loss': metrics['pg_loss'],
                'train/vf_loss': metrics['vf_loss'],
                'train/ent_coef': args.ent_coef,
            }
            if method == 'cchain':
                log_dict['chain/p_reg_coef'] = chain_ctrl.coef
                log_dict['chain/p_reg_loss'] = float(metrics['chain_p_reg_loss'])
                log_dict['chain/policy_churn'] = float(metrics['policy_churn'])
                log_dict['chain/value_churn'] = float(metrics['value_churn'])
            for net_name, st in dormant.items():
                log_dict[f'dormant/{net_name}_count'] = st['dormant_count']
                log_dict[f'dormant/{net_name}_fraction'] = st['dormant_fraction']
                log_dict[f'dormant/{net_name}_zero_fraction'] = st['zero_fraction']
            wandb.log(log_dict)

            dormant_str = ''
            if dormant:
                dormant_str = ' | Dormant ' + ' '.join(
                    f"{n[0]}:{s['dormant_count']}/{s['total_neurons']}"
                    for n, s in dormant.items()
                )
            print(f"Update {update+1:5d} | Timestep {timestep:10,} | "
                  f"Reward: {mean_reward:8.2f} ± {std_reward:.2f} | Best: {best_reward:8.2f}"
                  f"{dormant_str}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s, Best (training): {best_reward:.2f}")
    
    # Final evaluation with 10 trials
    print(f"\nFinal evaluation (10 trials)...")
    final_eval_fn = make_eval_fn(
        env, env_params, policy_network, hp['episode_length'], 10
    )
    key, eval_key = random.split(key)
    final_eval_rewards = [float(r) for r in np.asarray(
        final_eval_fn(policy_state.params, eval_key)
    )]
    for eval_trial, trial_reward in enumerate(final_eval_rewards):
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
    ckpt_path = os.path.join(output_dir, f"{method}_{env_name.replace('-', '_')}_best.pkl")
    ckpt_data = {
        'policy_params': policy_state.params,
        'value_params': value_state.params,
        'best_reward': best_reward,
        'final_eval_mean': final_mean,
        'final_eval_std': final_std,
        'config': config,
    }
    if method == 'pt':
        ckpt_data['value_params_perm'] = value_state_p.params
    with open(ckpt_path, 'wb') as f:
        pickle.dump(ckpt_data, f)
    print(f"Saved: {ckpt_path}")
    
    # Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    # GIFs are rendered post-hoc from the saved checkpoint, not here -- see
    # scripts/neurips_2026_rebuttal/make_gifs.py. Rendering is host-side
    # matplotlib and has no business inside a training run.

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
