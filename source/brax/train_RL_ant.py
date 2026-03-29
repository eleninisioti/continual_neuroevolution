"""
Train PPO on Brax Ant (non-continual).

Uses the standard Brax Ant environment with default parameters.

Usage:
    python source/brax/train_RL_ant.py --gpus 0
    python source/brax/train_RL_ant.py --num_timesteps 50000000 --gpus 0
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

import functools
import json
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs import create
from brax.training.agents.ppo import networks as ppo_networks
import wandb

from my_brax.ppo_train import train as ppo_train


def parse_args():
    parser = argparse.ArgumentParser(description='PPO on Brax Ant (Non-Continual)')
    parser.add_argument('--env', type=str, default='ant',
                        help='Brax environment name')
    parser.add_argument('--num_timesteps', type=int, default=50_000_000,
                        help='Total timesteps')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)

    # PPO hyperparameters (from brax defaults for ant)
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--entropy_cost', type=float, default=1e-2)
    parser.add_argument('--discounting', type=float, default=0.97)
    parser.add_argument('--unroll_length', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--num_updates_per_batch', type=int, default=4)
    parser.add_argument('--normalize_observations', type=bool, default=True)
    parser.add_argument('--reward_scaling', type=float, default=10.0)
    parser.add_argument('--num_evals', type=int, default=10)
    parser.add_argument('--num_eval_envs', type=int, default=128)
    parser.add_argument('--action_repeat', type=int, default=1)

    # Method variants
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
    parser.add_argument('--wandb_project', type=str, default='brax_ant_ppo')
    parser.add_argument('--run_name', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    env_name = args.env
    num_timesteps = args.num_timesteps
    seed = args.seed
    trial = args.trial

    output_dir = args.output_dir or f"projects/brax/ppo_{env_name}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"PPO on Brax {env_name} (Non-Continual)")
    print("=" * 60)
    print(f"  Total timesteps: {num_timesteps:,}")
    print(f"  Seed: {seed}")
    print(f"  Output: {output_dir}")

    # Create environment
    env = create(env_name, episode_length=args.episode_length)

    print(f"  Obs size: {env.observation_size}, Action size: {env.action_size}")

    # Determine algorithm name
    if args.use_redo:
        algo_name = "redo_ppo"
    elif args.use_trac:
        algo_name = "trac_ppo"
    else:
        algo_name = "ppo"

    # Initialize wandb
    config = {
        'env': env_name, 'num_timesteps': num_timesteps,
        'seed': seed, 'trial': trial, 'algorithm': algo_name,
        'num_envs': args.num_envs, 'episode_length': args.episode_length,
        'learning_rate': args.learning_rate, 'entropy_cost': args.entropy_cost,
        'discounting': args.discounting, 'unroll_length': args.unroll_length,
        'batch_size': args.batch_size, 'num_minibatches': args.num_minibatches,
        'num_updates_per_batch': args.num_updates_per_batch,
        'reward_scaling': args.reward_scaling, 'action_repeat': args.action_repeat,
        'use_trac': args.use_trac, 'use_redo': args.use_redo,
    }
    run_name = args.run_name or f"{algo_name}_{env_name}_trial{trial}"
    wandb.init(project=args.wandb_project, config=config,
               name=run_name, reinit=True)

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
        policy_hidden_layer_sizes=(256, 256),
        value_hidden_layer_sizes=(256, 256),
    )

    start_time = time.time()

    # Train PPO
    make_inference_fn, params, metrics = ppo_train(
        environment=env,
        num_timesteps=num_timesteps,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        normalize_observations=args.normalize_observations,
        reward_scaling=args.reward_scaling,
        action_repeat=args.action_repeat,
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
    ckpt_path = os.path.join(output_dir, f"{algo_name}_{env_name}_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'params': params,
            'best_reward': best_reward,
            'config': config,
        }, f)
    print(f"Saved: {ckpt_path}")

    # Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
