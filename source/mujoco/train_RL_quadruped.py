"""
PPO training for Go1 quadruped with single leg damage (non-continual).

This trains for 2x the timesteps of a single task in the continual setting,
allowing fair comparison between continual and non-continual approaches.

Usage:
    python train_RL_quadruped.py --leg FR --cuda_device 0
    python train_RL_quadruped.py --leg FL --cuda_device 1
    python train_RL_quadruped.py --leg RR --cuda_device 2
    python train_RL_quadruped.py --leg RL --cuda_device 3
"""

import os
import sys

# Add repo root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Parse --cuda_device argument BEFORE importing JAX
def _get_cuda_device_arg():
    for i, arg in enumerate(sys.argv):
        if arg == '--cuda_device' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg == '--gpus' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

_cuda_device = _get_cuda_device_arg()
if _cuda_device:
    os.environ['CUDA_VISIBLE_DEVICES'] = _cuda_device
    print(f"Setting CUDA_VISIBLE_DEVICES={_cuda_device}")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import argparse
import functools
import json
import pickle
import time

import jax
import jax.numpy as jnp
import wandb
import imageio

from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground._src import wrapper

from my_brax.ppo_train import train as ppo_train


# Leg action indices
LEG_ACTION_INDICES = {
    'FR': [0, 1, 2],    # Front Right
    'FL': [3, 4, 5],    # Front Left
    'RR': [6, 7, 8],    # Rear Right
    'RL': [9, 10, 11],  # Rear Left
}

# Leg qpos indices (joint positions in state.data.qpos)
# qpos layout: [0:7] = base pos/quat, [7:19] = 12 joint angles
LEG_QPOS_INDICES = {
    'FR': [7, 8, 9],
    'FL': [10, 11, 12],
    'RR': [13, 14, 15],
    'RL': [16, 17, 18],
}

# Leg qvel indices (joint velocities in state.data.qvel)
# qvel layout: [0:6] = base lin/ang vel, [6:18] = 12 joint velocities
LEG_QVEL_INDICES = {
    'FR': [6, 7, 8],
    'FL': [9, 10, 11],
    'RR': [12, 13, 14],
    'RL': [15, 16, 17],
}

# Locked joint positions (bent/tucked position)
LOCKED_JOINT_POSITIONS = jnp.array([0.0, 1.2, -2.4])


class LegDamageWrapper:
    """Wrapper that locks a damaged leg in a fixed bent position.
    
    The damaged leg's joints are locked at a fixed position with zero velocity,
    simulating a completely broken/frozen limb that cannot move.
    """
    
    def __init__(self, env, damaged_leg):
        """
        Args:
            env: The base environment
            damaged_leg: Which leg to damage ('FR', 'FL', 'RR', 'RL')
        """
        self._env = env
        self._damaged_leg = damaged_leg
        
        # Create action mask: 1 for healthy, 0 for damaged
        self._action_mask = jnp.ones(env.action_size)
        if damaged_leg is not None:
            action_indices = jnp.array(LEG_ACTION_INDICES[damaged_leg])
            self._action_mask = self._action_mask.at[action_indices].set(0.0)
            
            # Store qpos/qvel indices for locking
            self._qpos_indices = jnp.array(LEG_QPOS_INDICES[damaged_leg])
            self._qvel_indices = jnp.array(LEG_QVEL_INDICES[damaged_leg])
        else:
            self._qpos_indices = None
            self._qvel_indices = None
    
    def __getattr__(self, name):
        """Forward all other attribute access to the wrapped environment."""
        return getattr(self._env, name)
    
    def _lock_leg_joints(self, state):
        """Lock the damaged leg's joints to fixed position with zero velocity."""
        if self._damaged_leg is None:
            return state
        
        # Lock joint positions to bent position
        new_qpos = state.data.qpos.at[self._qpos_indices].set(LOCKED_JOINT_POSITIONS)
        # Lock joint velocities to zero
        new_qvel = state.data.qvel.at[self._qvel_indices].set(0.0)
        
        # Update state with locked joints
        new_data = state.data.replace(qpos=new_qpos, qvel=new_qvel)
        return state.replace(data=new_data)
    
    def step(self, state, action):
        # Zero out the damaged leg's actions
        masked_action = action * self._action_mask
        # Take step
        next_state = self._env.step(state, masked_action)
        # Lock the damaged leg's joints after the step
        next_state = self._lock_leg_joints(next_state)
        return next_state
    
    def reset(self, rng):
        state = self._env.reset(rng)
        # Lock the damaged leg's joints on reset too
        return self._lock_leg_joints(state)


def parse_args():
    parser = argparse.ArgumentParser(description='PPO on Go1 quadruped (Non-Continual)')
    parser.add_argument('--env', type=str, default='Go1JoystickFlatTerrain')
    parser.add_argument('--leg', type=str, default='FR', choices=['FR', 'FL', 'RR', 'RL', 'NONE'],
                        help='Which leg to damage (NONE for healthy robot)')
    parser.add_argument('--num_timesteps', type=int, default=20_000_000,
                        help='Total timesteps (default 20M = 2x single task in continual)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--cuda_device', type=str, default='0')
    parser.add_argument('--gpus', type=str, default=None, help='Alias for cuda_device')
    
    # PPO hyperparameters (Go1 defaults - same as continual)
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--entropy_cost', type=float, default=1e-2)
    parser.add_argument('--discounting', type=float, default=0.97)
    parser.add_argument('--unroll_length', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--num_updates_per_batch', type=int, default=4)
    parser.add_argument('--normalize_observations', type=lambda x: x.lower() == 'true',
                        default=True, metavar='BOOL')
    parser.add_argument('--reward_scaling', type=float, default=1.0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--num_evals', type=int, default=100)
    parser.add_argument('--num_eval_envs', type=int, default=128)
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='mujoco_evosax')
    
    # TRAC optimizer
    parser.add_argument('--use_trac', action='store_true', default=False,
                        help='Use TRAC optimizer for adaptive learning rates')
    
    # ReDo (Reinitializing Dormant Neurons)
    parser.add_argument('--use_redo', action='store_true', default=False,
                        help='Use ReDo (Reinitializing Dormant Neurons)')
    parser.add_argument('--redo_frequency', type=int, default=10,
                        help='Apply ReDo every N epochs')
    parser.add_argument('--redo_tau', type=float, default=0.01,
                        help='Threshold for dormant neuron detection')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    env_name = args.env
    damaged_leg = args.leg if args.leg != 'NONE' else None
    leg_label = args.leg  # Keep original for naming
    num_timesteps = args.num_timesteps
    seed = args.seed
    trial = args.trial
    
    # Output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = f"projects/mujoco/ppo_{env_name}_leg{leg_label}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"PPO on {env_name} (Non-Continual)")
    print("=" * 60)
    print(f"  Damaged leg: {leg_label}")
    print(f"  Total timesteps: {num_timesteps:,}")
    print(f"  Seed: {seed}, Trial: {trial}")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    
    # Create environment with leg damage
    base_env = registry.load(env_name)
    env = LegDamageWrapper(base_env, damaged_leg)
    
    # Get env info
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)
    
    # Go1 returns dict observations with 'state' key
    if isinstance(state.obs, dict):
        obs_dim = state.obs['state'].shape[-1]
    else:
        obs_dim = state.obs.shape[-1]
    action_dim = env.action_size
    
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Save config
    if args.use_redo:
        algo_name = "redo_ppo"
    elif args.use_trac:
        algo_name = "trac_ppo"
    else:
        algo_name = "ppo"
    config = {
        'env': env_name,
        'damaged_leg': damaged_leg,
        'num_timesteps': num_timesteps,
        'seed': seed,
        'trial': trial,
        'num_envs': args.num_envs,
        'learning_rate': args.learning_rate,
        'entropy_cost': args.entropy_cost,
        'discounting': args.discounting,
        'normalize_observations': args.normalize_observations,
        'use_trac': args.use_trac,
        'use_redo': args.use_redo,
        'redo_frequency': args.redo_frequency,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=f"{algo_name}_{env_name}_leg{leg_label}_trial{trial}",
        config=config,
        reinit=True,
    )
    
    # Track metrics
    start_time = time.time()
    best_reward = -float('inf')
    training_metrics = []
    
    def progress_fn(step, metrics):
        nonlocal best_reward
        
        reward = float(metrics.get('eval/episode_reward', 0.0))
        reward_std = float(metrics.get('eval/episode_reward_std', 0.0))
        
        if reward > best_reward:
            best_reward = reward
        
        training_metrics.append({
            'step': int(step),
            'reward': reward,
            'reward_std': reward_std,
            'best_reward': best_reward,
        })
        
        wandb.log({
            'step': step,
            'eval/episode_reward': reward,
            'eval/episode_reward_std': reward_std,
            'best_reward': best_reward,
        })
        
        elapsed = time.time() - start_time
        print(f"Step {step:10,} | Reward: {reward:8.2f} ± {reward_std:5.2f} | "
              f"Best: {best_reward:8.2f} | Time: {elapsed:6.1f}s")
    
    # Network factory (larger network for quadruped - same as continual)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
    )
    
    # Wrap environment for training
    def wrap_env_fn(env, episode_length=1000, action_repeat=1, randomization_fn=None):
        return wrapper.wrap_for_brax_training(
            env,
            action_repeat=action_repeat,
            episode_length=episode_length,
            randomization_fn=randomization_fn,
        )
    
    # Train PPO
    make_inference_fn, params, metrics = ppo_train(
        environment=env,
        num_timesteps=num_timesteps,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        wrap_env_fn=wrap_env_fn,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        normalize_observations=args.normalize_observations,
        reward_scaling=args.reward_scaling,
        max_grad_norm=args.max_grad_norm,
        num_evals=args.num_evals,
        num_eval_envs=args.num_eval_envs,
        network_factory=network_factory,
        seed=seed,
        progress_fn=progress_fn,
        use_trac=args.use_trac,
        use_redo=args.use_redo,
        redo_frequency=args.redo_frequency,
        redo_tau=args.redo_tau,
    )
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s, Best: {best_reward:.2f}")
    
    # Save checkpoint
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{algo_name}_{env_name}_leg{leg_label}_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'normalizer_params': params[0],
            'policy_params': params[1],
            'value_params': params[2],
            'params': params,  # Also save in tuple format for compatibility
            'best_reward': best_reward,
            'damaged_leg': damaged_leg,
            'config': config,
        }, f)
    print(f"Saved checkpoint: {ckpt_path}")
    
    # Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    # Save GIFs - following the same pattern as continual version
    try:
        gifs_dir = os.path.join(output_dir, "gifs")
        os.makedirs(gifs_dir, exist_ok=True)
        
        num_gifs = 10
        max_steps = 1000
        
        # JIT compile functions (same as continual version)
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)
        inference_fn = make_inference_fn(params, deterministic=True)
        jit_inference_fn = jax.jit(inference_fn)
        
        print(f"\nSaving {num_gifs} evaluation GIFs...")
        
        for gif_idx in range(num_gifs):
            # Use random seeds for diverse trajectories
            gif_key = jax.random.key(gif_idx * 37 + seed)
            state = jit_reset(gif_key)
            rollout = [state]
            total_reward = 0.0
            
            for _ in range(max_steps):
                gif_key, act_key = jax.random.split(gif_key)
                # Pass full dict observation - normalizer expects dict format
                action, _ = jit_inference_fn(state.obs, act_key)
                state = jit_step(state, action)
                rollout.append(state)
                total_reward += float(state.reward)
                if state.done:
                    break
            
            # Render every 2nd frame for smaller GIFs
            images = env.render(rollout[::2], height=240, width=320, camera="track")
            gif_path = os.path.join(gifs_dir, f"trajectory_{gif_idx:02d}_reward{total_reward:.0f}.gif")
            imageio.mimsave(gif_path, images, fps=30, loop=0)
        
        print(f"Saved {num_gifs} GIFs to: {gifs_dir}")
        
    except Exception as e:
        print(f"Warning: Failed to save GIFs: {e}")
        import traceback
        traceback.print_exc()
    
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
