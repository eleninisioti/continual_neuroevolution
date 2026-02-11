"""
PPO continual training for Go1 quadruped with leg damage.

All tasks damage one leg (locks joints in fixed position).
When switching tasks, a different random leg is chosen for damage.
The damaged leg's joints are locked at a fixed position, simulating a broken/frozen limb.

Task switching: every 50 generations (50 × 512,000 = 25,600,000 steps)

Go1 action structure (12 actions total):
  FR (Front Right): indices 0-2 (hip, thigh, calf)
  FL (Front Left): indices 3-5 (hip, thigh, calf)
  RR (Rear Right): indices 6-8 (hip, thigh, calf)
  RL (Rear Left): indices 9-11 (hip, thigh, calf)
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
    return None

_cuda_device = _get_cuda_device_arg()
if _cuda_device:
    os.environ['CUDA_VISIBLE_DEVICES'] = _cuda_device
    print(f"Setting CUDA_VISIBLE_DEVICES={_cuda_device}")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
import pickle
import jax
import jax.numpy as jnp
from jax import random
from datetime import datetime
import numpy as np
import functools
import wandb
import imageio

from mujoco_playground import registry
from mujoco_playground._src import wrapper

# Import continual training function
from my_brax.ppo_continual_train import train_continual


# Leg action indices (None = healthy/no damage)
LEG_ACTION_INDICES = {
    None: [],        # Healthy (no damage)
    0: [0, 1, 2],    # Front Right (FR) - action indices
    1: [3, 4, 5],    # Front Left (FL) - action indices
    2: [6, 7, 8],    # Rear Right (RR) - action indices
    3: [9, 10, 11],  # Rear Left (RL) - action indices
}

# Leg qpos indices (joint positions in state.data.qpos)
# qpos layout: [0:7] = base pos/quat, [7:19] = 12 joint angles
LEG_QPOS_INDICES = {
    None: [],        # Healthy (no damage)
    0: [7, 8, 9],    # Front Right (FR) - qpos indices
    1: [10, 11, 12], # Front Left (FL) - qpos indices
    2: [13, 14, 15], # Rear Right (RR) - qpos indices
    3: [16, 17, 18], # Rear Left (RL) - qpos indices
}

# Leg qvel indices (joint velocities in state.data.qvel)
# qvel layout: [0:6] = base lin/ang vel, [6:18] = 12 joint velocities
LEG_QVEL_INDICES = {
    None: [],        # Healthy (no damage)
    0: [6, 7, 8],    # Front Right (FR) - qvel indices
    1: [9, 10, 11],  # Front Left (FL) - qvel indices
    2: [12, 13, 14], # Rear Right (RR) - qvel indices
    3: [15, 16, 17], # Rear Left (RL) - qvel indices
}

# Locked joint positions (bent/tucked position)
LOCKED_JOINT_POSITIONS = jnp.array([0.0, 1.2, -2.4])  # hip=0, thigh bent up, calf bent back

LEG_NAMES = {None: 'HEALTHY', 0: 'FR', 1: 'FL', 2: 'RR', 3: 'RL'}


class LegDamageWrapper:
    """Wrapper that locks a damaged leg in a fixed bent position.
    
    The damaged leg's joints are locked at a fixed position with zero velocity,
    simulating a completely broken/frozen limb that cannot move.
    """
    
    def __init__(self, env, damaged_leg_idx):
        """
        Args:
            env: The base environment
            damaged_leg_idx: Which leg to damage (0=FR, 1=FL, 2=RR, 3=RL), or None for healthy
        """
        self._env = env
        self._damaged_leg = damaged_leg_idx
        
        # Create action mask: 1 for healthy, 0 for damaged
        self._action_mask = jnp.ones(env.action_size)
        if damaged_leg_idx is not None:
            action_indices = jnp.array(LEG_ACTION_INDICES[damaged_leg_idx])
            self._action_mask = self._action_mask.at[action_indices].set(0.0)
            
            # Store qpos/qvel indices for locking
            self._qpos_indices = jnp.array(LEG_QPOS_INDICES[damaged_leg_idx])
            self._qvel_indices = jnp.array(LEG_QVEL_INDICES[damaged_leg_idx])
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


def create_leg_damage_env_factory(base_env_name: str, leg_sequence: list):
    """Create an environment factory for leg damage tasks.
    
    The factory maps task indices (0, 1, 2, ...) to damaged leg environments.
    
    Args:
        base_env_name: Base environment name (e.g., 'Go1JoystickFlatTerrain')
        leg_sequence: List of leg indices to damage for each task
    
    Returns:
        env_factory function compatible with train_continual
    """
    def env_factory(task_idx):
        """Create train and eval environments for a given task index.
        
        Args:
            task_idx: Integer task index (0, 1, 2, ...)
        
        Returns:
            Tuple of (train_env, eval_env)
        """
        # task_idx is passed as float from train_continual, convert to int
        task_idx = int(task_idx)
        damaged_leg = leg_sequence[task_idx]
        
        train_env = registry.load(base_env_name)
        train_env = LegDamageWrapper(train_env, damaged_leg)
        
        eval_env = registry.load(base_env_name)
        eval_env = LegDamageWrapper(eval_env, damaged_leg)
        
        return train_env, eval_env
    
    return env_factory


def wrap_env_for_training(env, episode_length=1000, action_repeat=1):
    """Wrap environment for Brax training."""
    return wrapper.wrap_for_brax_training(
        env,
        action_repeat=action_repeat,
        episode_length=episode_length,
    )


def generate_random_leg_sequence(rng_key, num_tasks: int, avoid_consecutive: bool = True):
    """Generate a random sequence of legs to damage.
    
    All tasks damage a random leg.
    
    Args:
        rng_key: JAX random key
        num_tasks: Number of tasks
        avoid_consecutive: If True, avoid damaging same leg twice in a row
    
    Returns:
        List of leg indices (0-3 for damaged legs)
    """
    sequence = []
    
    for i in range(num_tasks):
        rng_key, subkey = random.split(rng_key)
        
        if avoid_consecutive and len(sequence) > 0:
            last_leg = sequence[-1]
            # Choose from legs other than the last one
            available_legs = [leg for leg in range(4) if leg != last_leg]
            idx = int(random.randint(subkey, (), 0, len(available_legs)))
            leg = available_legs[idx]
        else:
            # First task or no restriction
            leg = int(random.randint(subkey, (), 0, 4))
        
        sequence.append(leg)
    
    return sequence


def main():
    parser = argparse.ArgumentParser(description='PPO Continual Training with Leg Damage')
    parser.add_argument('--env', type=str, default='Go1JoystickFlatTerrain',
                        help='Base environment name')
    parser.add_argument('--num_tasks', type=int, default=20,
                        help='Number of leg damage tasks')
    parser.add_argument('--num_timesteps_per_task', type=int, default=10_000_000,
                        help='Training timesteps per task')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='projects',
                        help='Output directory')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--cuda_device', type=str, default='0',
                        help='CUDA device to use')
    parser.add_argument('--avoid_consecutive', action='store_true', default=True,
                        help='Avoid damaging same leg twice in a row')
    parser.add_argument('--trial', type=int, default=1,
                        help='Trial number for multiple runs')
    parser.add_argument('--use_trac', action='store_true', default=False,
                        help='Use TRAC optimizer')
    parser.add_argument('--use_redo', action='store_true', default=False,
                        help='Use ReDo (Reinitializing Dormant Neurons)')
    parser.add_argument('--redo_frequency', type=int, default=10,
                        help='Apply ReDo every N epochs')
    parser.add_argument('--normalize_observations', type=lambda x: x.lower() == 'true',
                        default=True, metavar='BOOL',
                        help='Normalize observations: true/false (default: true). '
                             'Disable for fair comparison with GA/ES in continual learning.')
    parser.add_argument('--track_dormant', action='store_true', default=False,
                        help='Track dormant neurons and their age')
    parser.add_argument('--dormant_tau', type=float, default=0.01,
                        help='Threshold for dormant neuron detection (default 0.01 = 1%% of layer mean)')
    parser.add_argument('--wandb_project', type=str, default='mujoco_evosax',
                        help='Wandb project name')
    args = parser.parse_args()
    
    # Generate random leg damage sequence
    rng_key = random.PRNGKey(args.seed + args.trial)
    leg_sequence = generate_random_leg_sequence(
        rng_key, 
        args.num_tasks, 
        avoid_consecutive=args.avoid_consecutive
    )
    
    print(f"\n{'='*60}")
    print(f"PPO Continual Training - Leg Damage Experiment")
    print(f"{'='*60}")
    print(f"Base Environment: {args.env}")
    print(f"Number of Tasks: {args.num_tasks}")
    print(f"Timesteps per Task: {args.num_timesteps_per_task:,} (50 generations × 512,000 steps)")
    print(f"Seed: {args.seed}, Trial: {args.trial}")
    print(f"CUDA Device: {args.cuda_device}")
    print(f"Use TRAC: {args.use_trac}")
    print(f"Use ReDo: {args.use_redo}" + (f" (frequency={args.redo_frequency})" if args.use_redo else ""))
    print(f"\nLeg Damage Sequence:")
    for i, leg_idx in enumerate(leg_sequence):
        if leg_idx is None:
            print(f"  Task {i+1}: HEALTHY (no damage)")
        else:
            print(f"  Task {i+1}: Damage {LEG_NAMES[leg_idx]} leg (action indices {LEG_ACTION_INDICES[leg_idx]})")
    print(f"{'='*60}\n")
    
    # Create experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"ppo_go1_legdamage_trial{args.trial}_{timestamp}"
    
    output_path = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_path, exist_ok=True)
    
    # Save experiment config
    config = {
        'env': args.env,
        'num_tasks': args.num_tasks,
        'num_timesteps_per_task': args.num_timesteps_per_task,
        'generations_per_task': args.num_timesteps_per_task // 512000,  # 50 generations
        'steps_per_generation': 512000,
        'seed': args.seed,
        'trial': args.trial,
        'leg_sequence': [l if l is not None else 'HEALTHY' for l in leg_sequence],
        'leg_names': [LEG_NAMES[i] for i in leg_sequence],
        'leg_action_indices': {LEG_NAMES[k]: v for k, v in LEG_ACTION_INDICES.items()},
        'leg_qpos_indices': {LEG_NAMES[k]: list(v) for k, v in LEG_QPOS_INDICES.items() if k is not None},
        'avoid_consecutive': args.avoid_consecutive,
        'use_trac': args.use_trac,
        'first_task_healthy': False,
    }
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize wandb
    trac_tag = "_trac" if args.use_trac else ""
    wandb.init(
        project=args.wandb_project + trac_tag,
        name=f"ppo_go1_legdamage_trial{args.trial}",
        config=config,
    )
    
    # Create environment factory
    env_factory = create_leg_damage_env_factory(args.env, leg_sequence)
    
    # Use task indices as "multipliers" - the env_factory maps these to damaged legs
    task_multipliers = list(range(args.num_tasks))
    
    # Progress callback with fitness tracking
    import time
    start_time = time.time()
    best_reward_overall = -float('inf')
    best_reward_per_task = {}
    results = []
    
    def progress_fn(global_step, task_idx, multiplier, metrics):
        nonlocal best_reward_overall
        
        reward = metrics.get('eval/episode_reward', 0)
        reward_std = metrics.get('eval/episode_reward_std', 0)
        generation = metrics.get('generation', 0)
        task_num = int(multiplier) + 1
        damaged_leg_idx = leg_sequence[int(multiplier)]
        damaged_leg = LEG_NAMES[damaged_leg_idx]
        
        # Log to wandb
        log_data = {
            'eval/episode_reward': reward,
            'eval/episode_reward_std': reward_std,
            'task': task_num,
            'damaged_leg_idx': int(multiplier) if damaged_leg_idx is not None else -1,
            'global_step': global_step,
            'generation': generation,
        }
        log_data.update(metrics)
        
        if 'eval/episode_reward' in metrics:
            # Track best rewards
            if task_num not in best_reward_per_task:
                best_reward_per_task[task_num] = -float('inf')
            if reward > best_reward_per_task[task_num]:
                best_reward_per_task[task_num] = reward
            if reward > best_reward_overall:
                best_reward_overall = reward
            
            # Add fitness metrics for consistency with GA
            log_data['fitness/best'] = float(reward)
            log_data['fitness/best_task'] = float(best_reward_per_task[task_num])
            log_data['fitness/best_overall'] = float(best_reward_overall)
            
            elapsed = time.time() - start_time
            leg_status = "HEALTHY" if damaged_leg_idx is None else f"{damaged_leg} damaged"
            print(f"Task {task_num} ({leg_status}), Step {global_step:>10,} | "
                  f"Reward: {reward:8.2f} ± {reward_std:5.2f} | "
                  f"Best: {best_reward_overall:8.2f} | "
                  f"Time: {elapsed:6.1f}s")
        
        wandb.log(log_data, step=generation)
        
        # Store result
        results.append({
            'task_idx': task_num,
            'damaged_leg': damaged_leg,
            'global_step': global_step,
            'reward': float(reward),
            'best_task': float(best_reward_per_task.get(task_num, reward)),
            'best_overall': float(best_reward_overall),
            'generation': generation,
        })
        
        # Save intermediate results
        with open(os.path.join(output_path, 'progress.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_path, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Checkpoint callback - save params at end of each task
    def checkpoint_fn(task_idx, params_dict):
        """Save checkpoint at end of each task."""
        leg_idx = leg_sequence[task_idx]
        leg_name = LEG_NAMES[leg_idx]
        
        ckpt_path = os.path.join(checkpoint_dir, f'task_{task_idx:02d}_{leg_name}.pkl')
        with open(ckpt_path, 'wb') as f:
            pickle.dump({
                'normalizer_params': params_dict['normalizer_params'],
                'policy_params': params_dict['policy_params'],
                'value_params': params_dict['value_params'],
                'task_idx': task_idx,
                'damaged_leg': leg_name,
                'damaged_leg_idx': leg_idx,
                'global_step': params_dict['global_step'],
            }, f)
        print(f"  Saved checkpoint: {ckpt_path}")
    
    # GIF callback function - saves evaluation GIFs at end of each task
    gifs_dir = os.path.join(output_path, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)
    
    def gif_callback_fn(task_idx, multiplier, inference_fn, env, params):
        """Save evaluation GIFs at end of task before switching."""
        try:
            num_gifs = 10  # Save 10 random trajectory GIFs per task
            max_steps = 1000
            
            jit_reset = jax.jit(env.reset)
            jit_step = jax.jit(env.step)
            jit_inference_fn = jax.jit(inference_fn)
            
            leg_idx = leg_sequence[task_idx]
            leg_name = LEG_NAMES[leg_idx]
            
            # Create task-specific subdirectory
            task_gifs_dir = os.path.join(gifs_dir, f"task_{task_idx:02d}_{leg_name}")
            os.makedirs(task_gifs_dir, exist_ok=True)
            
            for gif_idx in range(num_gifs):
                # Use random seeds for diverse trajectories
                gif_key = jax.random.key(task_idx * 1000 + gif_idx * 37 + args.seed)
                state = jit_reset(gif_key)
                rollout = [state]
                total_reward = 0.0
                
                for _ in range(max_steps):
                    gif_key, act_key = jax.random.split(gif_key)
                    action, _ = jit_inference_fn(state.obs, act_key)
                    state = jit_step(state, action)
                    rollout.append(state)
                    total_reward += float(state.reward)
                    if state.done:
                        break
                
                # Render every 2nd frame for smaller GIFs
                images = env.render(rollout[::2], height=240, width=320, camera="tracking")
                gif_path = os.path.join(task_gifs_dir, f"trajectory_{gif_idx:02d}_reward{total_reward:.0f}.gif")
                imageio.mimsave(gif_path, images, fps=30, loop=0)
            
            print(f"  Saved {num_gifs} GIFs for task {task_idx} in {task_gifs_dir}")
            
        except Exception as e:
            print(f"  Warning: Failed to save GIFs for task {task_idx}: {e}")
    
    # Run continual training
    make_policy, final_params, final_metrics = train_continual(
        env_factory=env_factory,
        task_multipliers=task_multipliers,
        timesteps_per_task=args.num_timesteps_per_task,
        # PPO hyperparameters (Go1 defaults)
        num_envs=4096,
        episode_length=1000,
        action_repeat=1,
        wrap_env_fn=wrap_env_for_training,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.97,
        unroll_length=20,
        batch_size=2048,
        num_minibatches=32,
        num_updates_per_batch=4,
        normalize_observations=args.normalize_observations,
        reward_scaling=1.0,
        max_grad_norm=1.0,
        seed=args.seed,
        num_evals_per_task=50,
        use_trac=args.use_trac,
        use_redo=args.use_redo,
        redo_frequency=args.redo_frequency,
        track_dormant=args.track_dormant,
        dormant_tau=args.dormant_tau,
        progress_fn=progress_fn,
        checkpoint_fn=checkpoint_fn,
        gif_callback_fn=gif_callback_fn,
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    
    # Compute per-task final rewards
    final_rewards = {}
    for r in results:
        task_idx = r['task_idx']
        final_rewards[task_idx] = r['reward']  # Last one wins
    
    print(f"\nFinal rewards by task:")
    for task_idx, reward in sorted(final_rewards.items()):
        leg_idx = leg_sequence[task_idx - 1]
        leg = LEG_NAMES[leg_idx]
        status = "HEALTHY" if leg_idx is None else f"{leg} damaged"
        print(f"  Task {task_idx}: {status} -> reward: {reward:.2f}")
    
    avg_reward = sum(final_rewards.values()) / len(final_rewards)
    print(f"\nAverage final reward: {avg_reward:.2f}")
    print(f"Results saved to: {output_path}")
    
    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main()