"""
Train PPO on CheetahRun with a single friction value (non-continual).

This trains for 2x the timesteps of a single task in the continual setting.

Usage:
    python train_RL_cheetah.py --friction 0.5 --gpus 0
    python train_RL_cheetah.py --friction 1.0 --gpus 0
    python train_RL_cheetah.py --friction 1.5 --gpus 0
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

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import functools
import json
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco import mjx
import wandb
import imageio

from my_brax.ppo_train import train as ppo_train


def create_env_with_friction(env_name, friction_mult):
    """Create environment with modified friction."""
    env = registry.load(env_name)
    env.mj_model.geom_friction[:] *= friction_mult
    env._mjx_model = mjx.put_model(env.mj_model)
    return env


def parse_args():
    parser = argparse.ArgumentParser(description='PPO on CheetahRun (Non-Continual)')
    parser.add_argument('--env', type=str, default='CheetahRun')
    parser.add_argument('--friction', type=float, default=1.0,
                        help='Friction multiplier (0.5, 1.0, or 1.5)')
    parser.add_argument('--num_timesteps', type=int, default=102_400_000,
                        help='Total timesteps (default 102.4M = 2x continual task)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)
    
    # PPO hyperparameters
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--entropy_cost', type=float, default=1e-2)
    parser.add_argument('--discounting', type=float, default=0.97)
    parser.add_argument('--unroll_length', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--num_updates_per_batch', type=int, default=4)
    parser.add_argument('--normalize_observations', type=bool, default=True)
    parser.add_argument('--reward_scaling', type=float, default=0.1)
    parser.add_argument('--num_evals', type=int, default=100)
    parser.add_argument('--num_eval_envs', type=int, default=128)
    parser.add_argument('--use_trac', action='store_true', default=False,
                        help='Use TRAC optimizer for adaptive learning rates')
    parser.add_argument('--use_redo', action='store_true', default=False,
                        help='Use ReDo (Reinitializing Dormant Neurons)')
    parser.add_argument('--redo_frequency', type=int, default=10,
                        help='Apply ReDo every N epochs')
    parser.add_argument('--redo_tau', type=float, default=0.01,
                        help='Threshold for dormant neuron detection')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='continual_neuroevolution_ppo')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    env_name = args.env
    friction = args.friction
    num_timesteps = args.num_timesteps
    seed = args.seed
    trial = args.trial
    
    friction_label = f"friction{friction:.1f}".replace(".", "p")
    
    output_dir = args.output_dir or f"projects/mujoco/ppo_{env_name}_{friction_label}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"PPO on {env_name} (Non-Continual)")
    print("=" * 60)
    print(f"  Friction multiplier: {friction}")
    print(f"  Total timesteps: {num_timesteps:,}")
    print(f"  Output: {output_dir}")
    
    # Create environment with friction
    env = create_env_with_friction(env_name, friction)
    
    # Get env info
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)
    obs_dim = state.obs.shape[-1]
    action_dim = env.action_size
    
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Initialize wandb
    if args.use_redo:
        algo_name = "redo_ppo"
    elif args.use_trac:
        algo_name = "trac_ppo"
    else:
        algo_name = "ppo"
    config = {
        'env': env_name, 'friction': friction, 'num_timesteps': num_timesteps,
        'seed': seed, 'trial': trial, 'num_envs': args.num_envs,
        'learning_rate': args.learning_rate, 'entropy_cost': args.entropy_cost,
        'use_trac': args.use_trac,
        'use_redo': args.use_redo,
        'redo_frequency': args.redo_frequency,
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"{algo_name}_{env_name}_{friction_label}_trial{trial}", reinit=True)
    
    # Track metrics
    best_reward = -float('inf')
    training_metrics = []
    
    def progress_fn(step, metrics):
        nonlocal best_reward
        
        reward = float(metrics.get('eval/episode_reward', 0.0))
        if reward > best_reward:
            best_reward = reward
        
        training_metrics.append({
            'step': int(step),
            'reward': reward,
            'best_reward': best_reward,
        })
        
        wandb.log({
            'step': step,
            'eval/episode_reward': reward,
            'best_reward': best_reward,
        })
        
        print(f"Step {step:10,} | Reward: {reward:8.2f} | Best: {best_reward:8.2f}")
    
    # Network factory
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128),
        value_hidden_layer_sizes=(128, 128),
    )
    
    start_time = time.time()
    
    # Train PPO
    make_inference_fn, params, metrics = ppo_train(
        environment=env,
        num_timesteps=num_timesteps,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        normalize_observations=args.normalize_observations,
        reward_scaling=args.reward_scaling,
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
    ckpt_path = os.path.join(output_dir, f"{algo_name}_{env_name}_{friction_label}_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'params': params,
            'best_reward': best_reward,
            'friction': friction,
            'config': config,
        }, f)
    print(f"Saved: {ckpt_path}")
    
    # Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    # Save GIFs
    try:
        gifs_dir = os.path.join(output_dir, "gifs")
        os.makedirs(gifs_dir, exist_ok=True)
        
        num_gifs = 3
        max_steps = args.episode_length
        
        # JIT compile functions for fast rollouts
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)
        inference_fn = make_inference_fn(params, deterministic=True)
        jit_inference_fn = jax.jit(inference_fn)
        
        print(f"\nSaving {num_gifs} evaluation GIFs...")
        
        for gif_idx in range(num_gifs):
            gif_key = jax.random.key(gif_idx * 37 + seed)
            state = jit_reset(gif_key)
            trajectory = [state]
            total_reward = 0.0
            
            for _ in range(max_steps):
                gif_key, act_key = jax.random.split(gif_key)
                action, _ = jit_inference_fn(state.obs, act_key)
                state = jit_step(state, action)
                trajectory.append(state)
                total_reward += float(state.reward)
                if state.done:
                    break
            
            images = env.render(trajectory[::2], height=240, width=320, camera="side")
            gif_path = os.path.join(gifs_dir, f"trial{gif_idx}_reward{total_reward:.0f}.gif")
            imageio.mimsave(gif_path, images, fps=30, loop=0)
        
        print(f"Saved {num_gifs} GIFs to: {gifs_dir}")
    except Exception as e:
        print(f"Warning: Failed to save GIFs: {e}")
    
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
