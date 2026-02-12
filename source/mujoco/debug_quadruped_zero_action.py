"""
Debug script for Go1 quadruped - apply zero actions and observe behavior.

This helps debug reward issues where the robot gets high rewards without moving.

Usage:
    python debug_quadruped_zero_action.py --leg FR --gpus 0
    python debug_quadruped_zero_action.py --leg None --gpus 0  # No damage
"""

import argparse
import os
import sys

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

os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jnp
from mujoco_playground import registry
import numpy as np
import imageio


# Leg damage configuration
LEG_ACTION_INDICES = {
    'FR': [0, 1, 2], 'FL': [3, 4, 5], 'RR': [6, 7, 8], 'RL': [9, 10, 11],
}
LEG_QPOS_INDICES = {
    'FR': [7, 8, 9], 'FL': [10, 11, 12], 'RR': [13, 14, 15], 'RL': [16, 17, 18],
}
LEG_QVEL_INDICES = {
    'FR': [6, 7, 8], 'FL': [9, 10, 11], 'RR': [12, 13, 14], 'RL': [15, 16, 17],
}
LOCKED_JOINT_POSITIONS = jnp.array([0.0, 1.2, -2.4])


class LegDamageWrapper:
    """Wrapper that locks a damaged leg in a fixed bent position."""
    
    def __init__(self, env, damaged_leg):
        self._env = env
        self._damaged_leg = damaged_leg
        
        self._action_mask = jnp.ones(env.action_size)
        if damaged_leg is not None:
            action_indices = jnp.array(LEG_ACTION_INDICES[damaged_leg])
            self._action_mask = self._action_mask.at[action_indices].set(0.0)
            self._qpos_indices = jnp.array(LEG_QPOS_INDICES[damaged_leg])
            self._qvel_indices = jnp.array(LEG_QVEL_INDICES[damaged_leg])
        else:
            self._qpos_indices = None
            self._qvel_indices = None
    
    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def _lock_leg_joints(self, state):
        if self._damaged_leg is None:
            return state
        new_qpos = state.data.qpos.at[self._qpos_indices].set(LOCKED_JOINT_POSITIONS)
        new_qvel = state.data.qvel.at[self._qvel_indices].set(0.0)
        new_data = state.data.replace(qpos=new_qpos, qvel=new_qvel)
        return state.replace(data=new_data)
    
    def step(self, state, action):
        masked_action = action * self._action_mask
        next_state = self._env.step(state, masked_action)
        return self._lock_leg_joints(next_state)
    
    def reset(self, rng):
        state = self._env.reset(rng)
        return self._lock_leg_joints(state)


def parse_args():
    parser = argparse.ArgumentParser(description='Debug Go1 Quadruped - Zero Action Test')
    parser.add_argument('--leg', type=str, default='None',
                        choices=['FR', 'FL', 'RR', 'RL', 'None'],
                        help='Damaged leg (FR, FL, RR, RL) or None for no damage')
    parser.add_argument('--episode_length', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='debug_output')
    parser.add_argument('--action_type', type=str, default='zero',
                        choices=['zero', 'random', 'standing'],
                        help='Type of action to apply')
    return parser.parse_args()


def main():
    args = parse_args()
    
    damaged_leg = None if args.leg == 'None' else args.leg
    episode_length = args.episode_length
    seed = args.seed
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n=== Debug Go1 Quadruped ===")
    print(f"Damaged leg: {damaged_leg}")
    print(f"Episode length: {episode_length}")
    print(f"Action type: {args.action_type}")
    print(f"Seed: {seed}")
    
    # Create environment
    base_env = registry.load('Go1JoystickFlatTerrain')
    env = LegDamageWrapper(base_env, damaged_leg)
    
    print(f"\nEnvironment info:")
    print(f"  Action size: {env.action_size}")
    print(f"  Observation shape: {env.observation_size}")
    
    # Initialize
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    
    # JIT compile
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    state = jit_reset(reset_key)
    
    # Define action based on type
    if args.action_type == 'zero':
        action = jnp.zeros(env.action_size)
        action_desc = "Zero action (no motor commands)"
    elif args.action_type == 'standing':
        # Default standing pose for Go1
        action = jnp.array([0.0, 0.8, -1.5] * 4)  # hip, thigh, calf for each leg
        action_desc = "Standing pose action"
    else:  # random
        key, action_key = jax.random.split(key)
        action = jax.random.uniform(action_key, (env.action_size,), minval=-1, maxval=1)
        action_desc = "Random action"
    
    print(f"\nAction: {action_desc}")
    print(f"Action values: {action}")
    
    # Run episode
    print(f"\nRunning episode...")
    
    total_reward = 0.0
    rewards = []
    states = []
    done_step = None
    
    for step in range(episode_length):
        states.append(state)
        
        # Step with the fixed action
        state = jit_step(state, action)
        reward = float(state.reward)
        done = bool(state.done)
        
        rewards.append(reward)
        total_reward += reward
        
        if done and done_step is None:
            done_step = step
            print(f"  Episode done at step {step}")
        
        # Print periodic info
        if step % 100 == 0 or step == episode_length - 1:
            # Get position info
            qpos = state.data.qpos
            body_height = float(qpos[2])  # z position of body
            body_x = float(qpos[0])
            body_y = float(qpos[1])
            
            # Get velocity
            qvel = state.data.qvel
            vx = float(qvel[0])
            vy = float(qvel[1])
            vz = float(qvel[2])
            
            print(f"  Step {step:4d}: reward={reward:7.3f}, cumulative={total_reward:8.2f}, "
                  f"pos=({body_x:.2f}, {body_y:.2f}, {body_height:.3f}), "
                  f"vel=({vx:.3f}, {vy:.3f}, {vz:.3f})")
    
    print(f"\n=== Results ===")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Mean reward per step: {total_reward / episode_length:.4f}")
    print(f"Max step reward: {max(rewards):.4f}")
    print(f"Min step reward: {min(rewards):.4f}")
    print(f"Done step: {done_step if done_step else 'Never'}")
    
    # Analyze reward components
    print(f"\n=== Reward Analysis ===")
    final_state = states[-1]
    qpos = final_state.data.qpos
    qvel = final_state.data.qvel
    
    print(f"Final body position: x={float(qpos[0]):.3f}, y={float(qpos[1]):.3f}, z={float(qpos[2]):.3f}")
    print(f"Final body velocity: vx={float(qvel[0]):.3f}, vy={float(qvel[1]):.3f}, vz={float(qvel[2]):.3f}")
    
    # Check if robot fell
    final_height = float(qpos[2])
    if final_height < 0.2:
        print(f"WARNING: Robot likely fell! Final height: {final_height:.3f}")
    
    # Render GIF
    print(f"\nRendering GIF...")
    
    # Sample frames for GIF (every N frames to keep file size reasonable)
    sample_rate = max(1, episode_length // 100)
    sampled_states = states[::sample_rate]
    
    frames = []
    for i, s in enumerate(sampled_states):
        frame = base_env.render(s, camera='track', width=480, height=360)
        frames.append(frame)
        if i % 10 == 0:
            print(f"  Rendered frame {i}/{len(sampled_states)}")
    
    # Save GIF
    gif_path = os.path.join(args.output_dir, f"debug_{args.action_type}_leg{args.leg}.gif")
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"\nGIF saved to: {gif_path}")
    
    # Save rewards data
    rewards_path = os.path.join(args.output_dir, f"debug_{args.action_type}_leg{args.leg}_rewards.txt")
    with open(rewards_path, 'w') as f:
        f.write(f"Total reward: {total_reward}\n")
        f.write(f"Episode length: {episode_length}\n")
        f.write(f"Action type: {args.action_type}\n")
        f.write(f"Damaged leg: {damaged_leg}\n")
        f.write(f"Done step: {done_step}\n")
        f.write(f"\nRewards per step:\n")
        for i, r in enumerate(rewards):
            f.write(f"{i}: {r}\n")
    print(f"Rewards saved to: {rewards_path}")
    
    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
