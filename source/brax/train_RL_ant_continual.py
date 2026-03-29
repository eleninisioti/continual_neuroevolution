"""
Train PPO on Brax Ant with Continual Learning (gravity changes).

This script trains across multiple tasks where gravity is modified between tasks.
The full training state (including optimizer) is preserved across tasks for true continual learning.

Usage:
    python source/brax/train_RL_ant_continual.py --gpus 0
    python source/brax/train_RL_ant_continual.py --num_tasks 30 --gpus 0
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

from my_brax.ppo_continual_train import train_continual


DEFAULT_GRAVITY_Z = -9.81


def create_ant_with_gravity(gravity_multiplier, episode_length=1000):
    """Create a Brax Ant environment with modified gravity.

    Args:
        gravity_multiplier: Multiplier for gravity (1.0 = normal, 0.2 = 1/5, 5.0 = 5x)
        episode_length: Max episode length
    Returns:
        Brax Ant environment with modified gravity
    """
    env = create('ant', episode_length=episode_length)
    gravity_z = DEFAULT_GRAVITY_Z * gravity_multiplier
    new_sys = env.unwrapped.sys.replace(gravity=jnp.array([0., 0., gravity_z]))
    env.unwrapped.sys = new_sys
    print(f"  Gravity set: multiplier={gravity_multiplier}, gravity_z={gravity_z:.2f}")
    return env


def sample_task_multipliers(num_tasks, low_mult=0.2, high_mult=5.0, seed=42):
    """Generate multiplier values for each task using random log-uniform sampling.

    Samples gravity multipliers uniformly in log-space between low_mult and high_mult.
    This follows the Dohare et al. setup where gravity is randomly sampled each task.
    
    Args:
        num_tasks: Number of tasks to sample multipliers for
        low_mult: Lower bound for gravity multiplier (e.g., 0.2 = 1/5 Earth gravity)
        high_mult: Upper bound for gravity multiplier (e.g., 5.0 = 5x Earth gravity)
        seed: Random seed for reproducibility
    """
    rng = np.random.RandomState(seed)
    # Log-uniform sampling: sample uniformly in log-space
    log_low = np.log(low_mult)
    log_high = np.log(high_mult)
    log_samples = rng.uniform(log_low, log_high, size=num_tasks)
    multipliers = np.exp(log_samples).tolist()
    return multipliers


def parse_args():
    parser = argparse.ArgumentParser(description='PPO Continual Learning on Brax Ant (gravity)')
    parser.add_argument('--num_tasks', type=int, default=30,
                        help='Number of tasks (default 30 = 10 repetitions of 3 gravity values)')
    parser.add_argument('--timesteps_per_task', type=int, default=50_000_000,
                        help='Timesteps per task (default 50M)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)

    # Gravity settings (random log-uniform sampling between low and high)
    parser.add_argument('--gravity_low_mult', type=float, default=0.2,
                        help='Low gravity multiplier bound (default 0.2 = 1/5 Earth gravity)')
    parser.add_argument('--gravity_high_mult', type=float, default=5.0,
                        help='High gravity multiplier bound (default 5.0 = 5x Earth gravity)')

    # PPO hyperparameters (from brax ant defaults / non-continual script)
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--num_eval_envs', type=int, default=128)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--entropy_cost', type=float, default=1e-2)
    parser.add_argument('--discounting', type=float, default=0.97)
    parser.add_argument('--unroll_length', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--num_updates_per_batch', type=int, default=4)
    parser.add_argument('--normalize_observations', type=lambda x: x.lower() == 'true',
                        default=True, metavar='BOOL')
    parser.add_argument('--reward_scaling', type=float, default=10.0)
    parser.add_argument('--clipping_epsilon', type=float, default=0.3)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--action_repeat', type=int, default=1)

    # Network architecture
    parser.add_argument('--policy_hidden_sizes', type=str, default='256,256',
                        help='Policy network hidden layer sizes (default "256,256")')
    parser.add_argument('--value_hidden_sizes', type=str, default='256,256',
                        help='Value network hidden layer sizes (default "256,256")')

    # Eval and logging
    parser.add_argument('--num_evals_per_task', type=int, default=100,
                        help='Number of evaluations per task during training')

    # Method variants
    parser.add_argument('--use_trac', action='store_true', default=False)
    parser.add_argument('--use_redo', action='store_true', default=False)
    parser.add_argument('--redo_frequency', type=int, default=10)
    parser.add_argument('--redo_tau', type=float, default=0.01)
    parser.add_argument('--track_dormant', action='store_true', default=False)
    parser.add_argument('--dormant_tau', type=float, default=0.01)

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='brax_ant_continual')
    parser.add_argument('--run_name', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse hidden layer sizes
    policy_hidden_sizes = tuple(int(x) for x in args.policy_hidden_sizes.split(','))
    value_hidden_sizes = tuple(int(x) for x in args.value_hidden_sizes.split(','))

    num_tasks = args.num_tasks
    timesteps_per_task = args.timesteps_per_task

    # Generate task multipliers (random log-uniform sampling)
    multipliers = sample_task_multipliers(
        num_tasks, args.gravity_low_mult, args.gravity_high_mult, seed=args.seed
    )

    # Determine algorithm name
    if args.use_redo:
        algo_name = "redo_ppo"
    elif args.use_trac:
        algo_name = "trac_ppo"
    else:
        algo_name = "ppo"

    # Output directory
    output_dir = args.output_dir or f"projects/brax/{algo_name}_ant_continual_gravity/trial_{args.trial}"
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    gen_checkpoint_dir = os.path.join(checkpoint_dir, "generations")
    os.makedirs(gen_checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("PPO Continual Learning on Brax Ant (gravity)")
    print("=" * 60)
    print(f"  Algorithm: {algo_name}")
    print(f"  Number of tasks: {num_tasks}")
    print(f"  Timesteps per task: {timesteps_per_task:,}")
    print(f"  Total timesteps: {num_tasks * timesteps_per_task:,}")
    print(f"  Gravity range: [{args.gravity_low_mult}x, {args.gravity_high_mult}x] (random log-uniform)")
    print(f"  Multiplier values: {[f'{m:.2f}' for m in multipliers]}")
    print(f"  Output: {output_dir}")

    print(f"\nPPO Hyperparameters:")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Unroll length: {args.unroll_length}")
    print(f"  Num minibatches: {args.num_minibatches}")
    print(f"  Num updates per batch: {args.num_updates_per_batch}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Entropy cost: {args.entropy_cost}")
    print(f"  Discounting: {args.discounting}")
    print(f"  Reward scaling: {args.reward_scaling}")
    print(f"  Policy hidden sizes: {policy_hidden_sizes}")
    print(f"  Value hidden sizes: {value_hidden_sizes}")
    print(f"  *** Optimizer state preserved across tasks ***")

    # Initialize wandb
    config = {
        "algorithm": f"{algo_name}-Continual",
        "env_name": "ant",
        "task_mod": "gravity",
        "num_tasks": num_tasks,
        "timesteps_per_task": timesteps_per_task,
        "total_timesteps": num_tasks * timesteps_per_task,
        "num_envs": args.num_envs,
        "num_eval_envs": args.num_eval_envs,
        "episode_length": args.episode_length,
        "unroll_length": args.unroll_length,
        "num_minibatches": args.num_minibatches,
        "num_updates_per_batch": args.num_updates_per_batch,
        "learning_rate": args.learning_rate,
        "entropy_cost": args.entropy_cost,
        "discounting": args.discounting,
        "reward_scaling": args.reward_scaling,
        "clipping_epsilon": args.clipping_epsilon,
        "gae_lambda": args.gae_lambda,
        "policy_hidden_sizes": policy_hidden_sizes,
        "value_hidden_sizes": value_hidden_sizes,
        "batch_size": args.batch_size,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "trial": args.trial,
        "gravity_sampling": "log_uniform",
        "gravity_low_mult": args.gravity_low_mult,
        "gravity_high_mult": args.gravity_high_mult,
        "task_multipliers": multipliers,
        "continual": True,
        "optimizer_preserved": True,
        "use_trac": args.use_trac,
        "use_redo": args.use_redo,
        "track_dormant": args.track_dormant,
        "output_dir": output_dir,
    }
    run_name = args.run_name or f"{algo_name}_ant_continual_gravity_trial{args.trial}"
    wandb.init(project=args.wandb_project, config=config,
               name=run_name, reinit=True)

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Network factory
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_hidden_sizes,
        value_hidden_layer_sizes=value_hidden_sizes,
    )

    # Environment factory: creates train and eval envs with given gravity multiplier
    def env_factory(multiplier):
        train_env = create_ant_with_gravity(multiplier, args.episode_length)
        eval_env = create_ant_with_gravity(multiplier, args.episode_length)
        return train_env, eval_env

    # Progress tracking
    start_time = time.time()
    best_reward_overall = -float('inf')
    best_reward_per_task = {}
    training_metrics_list = []

    def progress_fn(global_step, task_idx, multiplier, metrics):
        nonlocal best_reward_overall

        log_data = {
            "global_step": global_step,
            "task": task_idx,
            "multiplier": multiplier,
        }
        log_data.update(metrics)

        if 'eval/episode_reward' in metrics:
            reward = metrics['eval/episode_reward']

            if task_idx not in best_reward_per_task:
                best_reward_per_task[task_idx] = -float('inf')
            if reward > best_reward_per_task[task_idx]:
                best_reward_per_task[task_idx] = reward
            if reward > best_reward_overall:
                best_reward_overall = reward

            log_data['fitness/best'] = float(reward)
            log_data['fitness/best_task'] = float(best_reward_per_task[task_idx])
            log_data['fitness/best_overall'] = float(best_reward_overall)

            elapsed = time.time() - start_time
            print(f"Task {task_idx+1} Step {global_step:>10,} | "
                  f"Reward: {reward:8.2f} | "
                  f"Best Task: {best_reward_per_task[task_idx]:8.2f} | "
                  f"Best Overall: {best_reward_overall:8.2f} | "
                  f"gravity: {multiplier:6.2f} | "
                  f"Time: {elapsed:6.1f}s", flush=True)

            training_metrics_list.append({
                'step': global_step,
                'task': task_idx,
                'multiplier': multiplier,
                'reward': float(reward),
                'best_task_reward': float(best_reward_per_task[task_idx]),
                'best_overall_reward': float(best_reward_overall),
                'elapsed_time': elapsed,
            })

        generation = metrics.get('generation', 0)
        log_data['generation'] = generation
        wandb.log(log_data, step=generation)

    # Checkpoint callback
    def checkpoint_fn(task_idx, params_dict):
        multiplier = params_dict.get('multiplier', 0.0)
        task_label = f"gravity_{multiplier:.2f}".replace(".", "p")
        checkpoint_path = os.path.join(checkpoint_dir, f"task_{task_idx:02d}_{task_label}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'normalizer_params': params_dict.get('normalizer_params'),
                'policy_params': params_dict.get('policy_params'),
                'value_params': params_dict.get('value_params'),
                'task_idx': task_idx,
                'multiplier': multiplier,
                'global_step': params_dict.get('global_step'),
                'config': {
                    'env_name': 'ant',
                    'policy_hidden_sizes': policy_hidden_sizes,
                    'value_hidden_sizes': value_hidden_sizes,
                }
            }, f)
        print(f"  Checkpoint saved: {checkpoint_path}")

    # Generation checkpoint callback
    def generation_checkpoint_fn(generation, params_dict):
        checkpoint_path = os.path.join(gen_checkpoint_dir, f"gen_{generation:05d}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'normalizer_params': params_dict.get('normalizer_params'),
                'policy_params': params_dict.get('policy_params'),
                'value_params': params_dict.get('value_params'),
                'task_idx': params_dict.get('task_idx'),
                'multiplier': params_dict.get('multiplier'),
                'generation': generation,
                'global_step': params_dict.get('global_step'),
                'config': {
                    'env_name': 'ant',
                    'policy_hidden_sizes': policy_hidden_sizes,
                    'value_hidden_sizes': value_hidden_sizes,
                }
            }, f)
        print(f"  Generation checkpoint saved: {checkpoint_path}")

    # Run continual training (wrap_env_fn=None uses default brax training wrapper)
    make_inference_fn, params, final_metrics = train_continual(
        env_factory=env_factory,
        task_multipliers=multipliers,
        timesteps_per_task=timesteps_per_task,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        wrap_env_fn=None,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        normalize_observations=args.normalize_observations,
        reward_scaling=args.reward_scaling,
        clipping_epsilon=args.clipping_epsilon,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        network_factory=network_factory,
        seed=args.seed,
        num_eval_envs=args.num_eval_envs,
        num_evals_per_task=args.num_evals_per_task,
        use_trac=args.use_trac,
        use_redo=args.use_redo,
        redo_frequency=args.redo_frequency,
        redo_tau=args.redo_tau,
        track_dormant=args.track_dormant,
        dormant_tau=args.dormant_tau,
        progress_fn=progress_fn,
        checkpoint_fn=checkpoint_fn,
        generation_checkpoint_fn=generation_checkpoint_fn,
    )

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Continual Learning Complete!")
    print("=" * 60)
    print(f"  Environment: Brax Ant")
    print(f"  Total tasks: {num_tasks}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Best reward achieved: {best_reward_overall:.2f}")

    # Save final policy
    save_path = os.path.join(output_dir, f"best_ant_{algo_name}_continual_gravity_policy.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump({
            'params': params,
            'config': {
                'env_name': 'ant',
                'policy_hidden_sizes': policy_hidden_sizes,
                'value_hidden_sizes': value_hidden_sizes,
                'num_tasks': num_tasks,
                'task_multipliers': multipliers,
            }
        }, f)
    print(f"  Final policy saved to: {save_path}")

    # Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics_list, f, indent=2)
    print(f"  Training metrics saved to: {metrics_path}")

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
