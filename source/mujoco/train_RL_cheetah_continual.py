"""
Train a PPO agent on MuJoCo Playground with Continual Learning (gravity changes).

This script trains across multiple tasks where gravity is modified between tasks.
The full training state (including optimizer) is preserved across tasks for true continual learning.

Usage:
    python train_ppo_continual.py --env HumanoidWalk
    python train_ppo_continual.py --env CheetahRun --num_tasks 5
    python train_ppo_continual.py --env WalkerWalk --use_recommended --num_tasks 10
"""

import argparse
import os
import sys

# Add repo root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Parse --gpus argument BEFORE importing JAX to set CUDA_VISIBLE_DEVICES
def _get_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpus' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

_gpu_arg = _get_gpu_arg()
if _gpu_arg:
    os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_arg
    print(f"Setting CUDA_VISIBLE_DEVICES={_gpu_arg}")

# Set JAX memory settings
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Set MuJoCo to use EGL for headless rendering (no display required)
os.environ["MUJOCO_GL"] = "egl"

# Now import JAX and other libraries
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
from mujoco_playground.config import dm_control_suite_params
import wandb
import matplotlib.pyplot as plt
import imageio

# Import our custom continual training function
from ppo_continual_train import train_continual


def create_modified_env(env_name, task_mod='gravity', multiplier=1.0):
    """Create environment with modified physics (gravity or friction).
    
    Args:
        env_name: Name of the environment
        task_mod: 'gravity' or 'friction'
        multiplier: Multiplier for the physics parameter (1.0 = normal)
    """
    from mujoco import mjx
    
    # Load the environment normally
    env = registry.load(env_name)
    
    if task_mod == 'gravity':
        # Modify gravity (default is -9.81)
        gravity_z = -9.81 * multiplier
        env.mj_model.opt.gravity[2] = gravity_z
        print(f"  Gravity set: multiplier={multiplier}, gravity_z={gravity_z:.2f}")
    elif task_mod == 'friction':
        # Modify friction for all geoms (scale sliding, torsional, rolling friction)
        env.mj_model.geom_friction[:] *= multiplier
        print(f"  Friction set: multiplier={multiplier}")
    else:
        raise ValueError(f"Unknown task_mod: {task_mod}. Use 'gravity' or 'friction'.")
    
    # Re-create mjx_model from modified mj_model
    env._mjx_model = mjx.put_model(env.mj_model)
    
    return env


def create_env_with_gravity(env_name, gravity_z):
    """Create environment with modified gravity (legacy function)."""
    return create_modified_env(env_name, task_mod='gravity', multiplier=gravity_z / -9.81)


def sample_task_multipliers(num_tasks, default_mult=1.0, low_mult=0.2, high_mult=5.0):
    """Generate multiplier values for each task, cycling through three values.
    
    Cycles through:
    1. default_mult (1.0 = normal)
    2. low_mult (0.2 = reduced)
    3. high_mult (5.0 = increased)
    
    Returns array of multipliers.
    """
    multiplier_cycle = [default_mult, low_mult, high_mult]
    multipliers = jnp.array([multiplier_cycle[i % 3] for i in range(num_tasks)])
    return multipliers


def sample_gravity_values(key, num_tasks, default_mult=1.0, low_mult=0.2, high_mult=5.0):
    """Generate gravity values for each task, cycling through three values.
    
    Default Earth gravity is -9.81. We cycle through:
    1. default_mult (1.0 = normal gravity)
    2. low_mult (0.2 = 1/5 gravity)
    3. high_mult (5.0 = 5x gravity)
    """
    default_gravity = -9.81
    
    # Cycle through default, low, high multipliers
    multiplier_cycle = [default_mult, low_mult, high_mult]
    multipliers = jnp.array([multiplier_cycle[i % 3] for i in range(num_tasks)])
    gravities = default_gravity * multipliers
    return gravities


def parse_args():
    parser = argparse.ArgumentParser(description='PPO Continual Learning on MuJoCo Playground')
    parser.add_argument('--env', type=str, default='CheetahRun',
                        help='Environment name (e.g., HumanoidWalk, CheetahRun)')
    parser.add_argument('--num_tasks', type=int, default=30,
                        help='Number of tasks (default 30 = 10 repetitions of 3 gravity values)')
    parser.add_argument('--timesteps_per_task', type=int, default=51_200_000,
                        help='Timesteps per task (default 51.2M)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs to use (e.g., "0,3,4"). Default: all available')
    
    # Task modification type
    parser.add_argument('--task_mod', type=str, default='gravity', choices=['gravity', 'friction'],
                        help='Which physics parameter to modify: gravity or friction (default: gravity)')
    
    # Gravity settings (cycle: default -> low -> high)
    parser.add_argument('--gravity_default_mult', type=float, default=1.0,
                        help='Default gravity multiplier (default 1.0 = normal Earth gravity)')
    parser.add_argument('--gravity_low_mult', type=float, default=0.2,
                        help='Low gravity multiplier (default 0.2 = 1/5 Earth gravity)')
    parser.add_argument('--gravity_high_mult', type=float, default=5.0,
                        help='High gravity multiplier (default 5.0 = 5x Earth gravity)')
    
    # Friction settings (cycle: default -> low -> high)
    parser.add_argument('--friction_default_mult', type=float, default=1.0,
                        help='Default friction multiplier (default 1.0 = normal friction)')
    parser.add_argument('--friction_low_mult', type=float, default=0.2,
                        help='Low friction multiplier (default 0.2 = slippery/icy)')
    parser.add_argument('--friction_high_mult', type=float, default=5.0,
                        help='High friction multiplier (default 5.0 = sticky/rough)')
    
    # PPO hyperparameters
    parser.add_argument('--num_envs', type=int, default=2048,
                        help='Number of parallel environments (default 2048)')
    parser.add_argument('--num_eval_envs', type=int, default=128,
                        help='Number of evaluation environments (default 128)')
    parser.add_argument('--episode_length', type=int, default=1000,
                        help='Max episode length (default 1000)')
    parser.add_argument('--unroll_length', type=int, default=10,
                        help='Unroll length for PPO (default 10)')
    parser.add_argument('--num_minibatches', type=int, default=32,
                        help='Number of minibatches (default 32)')
    parser.add_argument('--num_updates_per_batch', type=int, default=8,
                        help='Number of PPO updates per batch (default 8)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate (default 3e-4)')
    parser.add_argument('--entropy_cost', type=float, default=1e-2,
                        help='Entropy bonus coefficient (default 1e-2)')
    parser.add_argument('--discounting', type=float, default=0.97,
                        help='Discount factor gamma (default 0.97)')
    parser.add_argument('--reward_scaling', type=float, default=0.1,
                        help='Reward scaling factor (default 0.1)')
    parser.add_argument('--clipping_epsilon', type=float, default=0.3,
                        help='PPO clipping epsilon (default 0.3)')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda (default 0.95)')
    parser.add_argument('--normalize_observations', type=lambda x: x.lower() == 'true',
                        default=True, metavar='BOOL',
                        help='Normalize observations: true/false (default: true). '
                             'Disable for fair comparison with GA/ES in continual learning.')
    parser.add_argument('--policy_hidden_sizes', type=str, default='32,32,32,32',
                        help='Policy network hidden layer sizes (default "32,32,32,32")')
    parser.add_argument('--value_hidden_sizes', type=str, default='32,32,32,32',
                        help='Value network hidden layer sizes (default "32,32,32,32")')
    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'swish'],
                        help='Activation function (default "tanh", brax default is "swish")')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (default 256)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (default 1.0)')
    parser.add_argument('--action_repeat', type=int, default=1,
                        help='Action repeat (default 1)')
    
    # Eval and logging
    parser.add_argument('--num_evals_per_task', type=int, default=100,
                        help='Number of evaluations per task during training (default 100 = 1 per GA generation)')
    
    # Output and logging
    parser.add_argument('--wandb_project', type=str, default='mujoco_evosax',
                        help='Wandb project name (default: mujoco_evosax)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Wandb run name. If None, auto-generated from config.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for saving results. If None, uses default naming.')
    parser.add_argument('--trac', action='store_true',
                        help='Enable Trac optimizer')
    parser.add_argument('--use_redo', action='store_true', default=False,
                        help='Use ReDo (Reinitializing Dormant Neurons)')
    parser.add_argument('--redo_frequency', type=int, default=10,
                        help='Apply ReDo every N epochs')
    parser.add_argument('--track_dormant', action='store_true', default=False,
                        help='Track dormant neurons and their age')
    parser.add_argument('--dormant_tau', type=float, default=0.01,
                        help='Threshold for dormant neuron detection (default 0.01 = 1%% of layer mean)')
    
    return parser.parse_args()


def final_evaluation(env, inference_fn, key, env_name, gravity,
                     num_eval_trials=100, max_steps=1000, save_dir="eval_results",
                     num_gifs=10):
    """Run final evaluation with GIF saving, JSON rewards, and histogram."""
    
    print("\n" + "=" * 60)
    print(f"Running Final Evaluation ({num_eval_trials} trials, gravity={gravity:.2f})...")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    gifs_dir = os.path.join(save_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)
    
    jit_inference_fn = jax.jit(inference_fn)
    
    def rollout_episode(key):
        key, reset_key = jax.random.split(key)
        
        def step_fn(carry, _):
            state, total_reward, done_flag, key = carry
            key, act_key = jax.random.split(key)
            action, _ = jit_inference_fn(state.obs, act_key)
            next_state = env.step(state, action)
            reward = next_state.reward
            done = next_state.done
            total_reward = total_reward + reward * (1.0 - done_flag)
            done_flag = jnp.maximum(done_flag, done)
            return (next_state, total_reward, done_flag, key), None
        
        state = env.reset(reset_key)
        (_, total_reward, _, _), _ = jax.lax.scan(
            step_fn, (state, 0.0, 0.0, key), None, length=max_steps
        )
        return total_reward
    
    keys = jax.random.split(key, num_eval_trials)
    
    @jax.jit
    def eval_all(keys):
        return jax.vmap(rollout_episode)(keys)
    
    print("  Running parallel evaluation on GPU...")
    rewards = eval_all(keys)
    rewards = np.array(rewards)
    print(f"  Evaluation complete! Mean reward: {np.mean(rewards):.2f}")
    
    # Save GIFs
    print(f"  Saving {num_gifs} GIFs...")
    gif_indices = np.linspace(0, num_eval_trials - 1, num_gifs, dtype=int)
    
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    for i, trial_idx in enumerate(gif_indices):
        reward_val = float(rewards[trial_idx])
        gif_key = jax.random.key(trial_idx)
        state = jit_reset(gif_key)
        rollout = [state]
        
        for _ in range(max_steps):
            gif_key, act_key = jax.random.split(gif_key)
            action, _ = jit_inference_fn(state.obs, act_key)
            state = jit_step(state, action)
            rollout.append(state)
            if state.done:
                break
        
        try:
            images = env.render(rollout[::2], height=240, width=320, camera="side")
            gif_path = os.path.join(gifs_dir, f"trial_{trial_idx:03d}_reward_{reward_val:.1f}.gif")
            imageio.mimsave(gif_path, images, fps=30, loop=0)
            print(f"    GIF {i+1}/{num_gifs}: trial {trial_idx}, reward = {reward_val:.2f}")
        except Exception as e:
            print(f"    GIF {i+1}/{num_gifs}: trial {trial_idx}, reward = {reward_val:.2f} (failed: {e})")
    
    # Save results
    rewards_list = rewards.tolist()
    rewards_data = {
        "env_name": env_name,
        "gravity": gravity,
        "num_trials": num_eval_trials,
        "rewards": rewards_list,
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
    }
    
    json_path = os.path.join(save_dir, "eval_rewards.json")
    with open(json_path, 'w') as f:
        json.dump(rewards_data, f, indent=2)
    print(f"\n  Rewards saved to: {json_path}")
    
    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(rewards):.2f}')
    plt.axvline(np.median(rewards), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(rewards):.2f}')
    plt.xlabel('Reward', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Final Evaluation - {env_name} (PPO Continual)\n({num_eval_trials} trials, gravity={gravity:.2f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    hist_path = os.path.join(save_dir, "reward_histogram.png")
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Histogram saved to: {hist_path}")
    
    # Log to wandb
    wandb.log({
        "eval/mean_reward": float(np.mean(rewards)),
        "eval/std_reward": float(np.std(rewards)),
        "eval/min_reward": float(np.min(rewards)),
        "eval/max_reward": float(np.max(rewards)),
    })
    wandb.log({"eval/histogram": wandb.Image(hist_path)})
    
    print(f"\n  Final Evaluation Results:")
    print(f"    Mean reward:   {np.mean(rewards):.2f}")
    print(f"    Std reward:    {np.std(rewards):.2f}")
    print(f"    Min reward:    {np.min(rewards):.2f}")
    print(f"    Max reward:    {np.max(rewards):.2f}")
    
    return rewards_list


def main():
    args = parse_args()
    env_name = args.env
    
    # Environment-specific hyperparameter defaults (from successful runs)
    env_specific_configs = {
        'CheetahRun': {
            'timesteps_per_task': 150_000_000,
            'num_envs': 4096,
            'learning_rate': 1e-4,
            'batch_size': 256,
            'discounting': 0.97,
            'entropy_cost': 0.01,
            'episode_length': 1000,
            'num_minibatches': 32,
            'num_updates_per_batch': 8,
            'unroll_length': 10,
            'reward_scaling': 0.1,
            'action_repeat': 1,
        },
        'WalkerWalk': {
            'timesteps_per_task': 150_000_000,
            'num_envs': 4096,
            'learning_rate': 1e-4,
            'batch_size': 256,
            'discounting': 0.97,
            'entropy_cost': 0.01,
            'episode_length': 1000,
            'num_minibatches': 32,
            'num_updates_per_batch': 8,
            'unroll_length': 10,
            'reward_scaling': 0.1,
            'action_repeat': 1,
        },
        'HumanoidWalk': {
            'timesteps_per_task': 50_000_000,
            'num_envs': 2048,
            'learning_rate': 3e-4,
            'batch_size': 1024,
            'discounting': 0.97,
            'entropy_cost': 1e-3,
            'episode_length': 1000,
            'num_minibatches': 32,
            'num_updates_per_batch': 8,
            'unroll_length': 10,
            'reward_scaling': 0.1,
            'action_repeat': 1,
            'policy_hidden_sizes': '128,128,128,128',
            'value_hidden_sizes': '128,128,128,128',
            'activation': 'swish',
        },
    }
    
    # Default values for checking if user explicitly set something
    param_defaults = {
        'timesteps_per_task': 5_000_000,
        'num_envs': 2048,
        'episode_length': 1000,
        'unroll_length': 10,
        'num_minibatches': 32,
        'num_updates_per_batch': 8,
        'learning_rate': 3e-4,
        'entropy_cost': 1e-2,
        'discounting': 0.97,
        'reward_scaling': 0.1,
        'batch_size': 256,
        'action_repeat': 1,
        'policy_hidden_sizes': '32,32,32,32',
        'value_hidden_sizes': '32,32,32,32',
        'activation': 'tanh',
    }
    
    # Apply environment-specific config (only for params not explicitly set by user)
    if env_name in env_specific_configs:
        env_config = env_specific_configs[env_name]
        print(f"\nApplying environment-specific hyperparameters for {env_name}:")
        for key, val in env_config.items():
            if hasattr(args, key):
                current_val = getattr(args, key)
                is_default = key in param_defaults and current_val == param_defaults[key]
                if is_default:
                    setattr(args, key, val)
                    print(f"  {key}: {val}")
                else:
                    print(f"  {key}: {current_val} (CLI override, env-specific: {val})")
    
    # Parse hidden layer sizes
    policy_hidden_sizes = tuple(int(x) for x in args.policy_hidden_sizes.split(','))
    value_hidden_sizes = tuple(int(x) for x in args.value_hidden_sizes.split(','))
    
    # Device info
    devices = jax.devices()
    num_devices = len(devices)
    
    num_tasks = args.num_tasks
    timesteps_per_task = args.timesteps_per_task
    gravity_default_mult = args.gravity_default_mult
    gravity_low_mult = args.gravity_low_mult
    gravity_high_mult = args.gravity_high_mult
    friction_default_mult = args.friction_default_mult
    friction_low_mult = args.friction_low_mult
    friction_high_mult = args.friction_high_mult
    task_mod = args.task_mod
    
    print("=" * 60)
    print(f"PPO Continual Learning on {env_name} (MuJoCo Playground/MJX)")
    print("=" * 60)
    print(f"\nUsing {num_devices} GPU(s): {[str(d) for d in devices]}")
    
    print(f"\nContinual Learning Setup:")
    print(f"  Task modification: {task_mod}")
    print(f"  Number of tasks: {num_tasks}")
    print(f"  Timesteps per task: {timesteps_per_task:,}")
    print(f"  Total timesteps: {num_tasks * timesteps_per_task:,}")
    
    # Sample task multipliers based on task_mod
    key = jax.random.key(args.seed)
    if task_mod == 'gravity':
        multipliers = sample_task_multipliers(num_tasks, gravity_default_mult, gravity_low_mult, gravity_high_mult)
        print(f"  Gravity multipliers: {gravity_default_mult}x -> {gravity_low_mult}x -> {gravity_high_mult}x (cycling)")
    else:  # friction
        multipliers = sample_task_multipliers(num_tasks, friction_default_mult, friction_low_mult, friction_high_mult)
        print(f"  Friction multipliers: {friction_default_mult}x -> {friction_low_mult}x -> {friction_high_mult}x (cycling)")
    
    multiplier_list = [float(m) for m in multipliers]
    print(f"  Multiplier values: {[f'{m:.2f}' for m in multiplier_list]}")
    
    # Determine output directory
    mod_suffix = "gravity" if task_mod == "gravity" else "friction"
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"results_ppo_continual_{env_name.lower()}_{mod_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Initialize wandb
    run_name = args.run_name if args.run_name else f"ppo_continual_{env_name}_{task_mod}_seed{args.seed}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "algorithm": "PPO-Continual",
            "env_name": env_name,
            "task_mod": task_mod,
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
            "gravity_default_mult": gravity_default_mult,
            "gravity_low_mult": gravity_low_mult,
            "gravity_high_mult": gravity_high_mult,
            "friction_default_mult": friction_default_mult,
            "friction_low_mult": friction_low_mult,
            "friction_high_mult": friction_high_mult,
            "task_multipliers": multiplier_list,
            "continual": True,
            "optimizer_preserved": True,
            "output_dir": output_dir,
            "track_dormant": args.track_dormant,
            "dormant_tau": args.dormant_tau,
            "use_redo": args.use_redo,
            "redo_frequency": args.redo_frequency,
        }
    )
    
    # Save config to JSON file
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(wandb.config.as_dict(), f, indent=2)
    print(f"Config saved to: {config_path}")
    
    # Initialize training metrics list for CSV
    training_metrics_list = []
    
    print("\nPPO Hyperparameters:")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Unroll length: {args.unroll_length}")
    print(f"  Num minibatches: {args.num_minibatches}")
    print(f"  Num updates per batch: {args.num_updates_per_batch}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Entropy cost: {args.entropy_cost}")
    print(f"  Discounting: {args.discounting}")
    print(f"  Reward scaling: {args.reward_scaling}")
    print(f"  Policy hidden sizes: {policy_hidden_sizes}")
    print(f"  Value hidden sizes: {value_hidden_sizes}")
    print(f"  Activation: {args.activation}")
    print(f"\n  *** Optimizer state preserved across tasks ***")
    
    # Get activation function
    activation_fn = jax.nn.swish if args.activation == 'swish' else jax.nn.tanh
    
    # Network factory
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_hidden_sizes,
        value_hidden_layer_sizes=value_hidden_sizes,
        activation=activation_fn,
    )
    
    # Environment factory
    def env_factory(multiplier):
        train_env = create_modified_env(env_name, task_mod, multiplier)
        eval_env = create_modified_env(env_name, task_mod, multiplier)
        return train_env, eval_env
    
    # Progress tracking
    start_time = time.time()
    best_reward_overall = -float('inf')
    best_reward_per_task = {}
    
    def progress_fn(global_step, task_idx, multiplier, metrics):
        nonlocal best_reward_overall
        
        # Log to wandb
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
            
            # Add fitness metrics for consistency with GA
            log_data['fitness/best'] = float(reward)
            log_data['fitness/best_task'] = float(best_reward_per_task[task_idx])
            log_data['fitness/best_overall'] = float(best_reward_overall)
            
            elapsed = time.time() - start_time
            print(f"Task {task_idx+1} Step {global_step:>10,} | "
                  f"Reward: {reward:8.2f} | "
                  f"Best Task: {best_reward_per_task[task_idx]:8.2f} | "
                  f"Best Overall: {best_reward_overall:8.2f} | "
                  f"{task_mod}: {multiplier:6.2f} | "
                  f"Time: {elapsed:6.1f}s")
            
            # Accumulate training metrics for CSV
            training_metrics_list.append({
                'step': global_step,
                'task': task_idx,
                'multiplier': multiplier,
                'reward': float(reward),
                'best_task_reward': float(best_reward_per_task[task_idx]),
                'best_overall_reward': float(best_reward_overall),
                'elapsed_time': elapsed,
            })
        
        # Use generation from metrics (computed as task_idx * num_evals_per_task + epoch)
        # This ensures tasks change exactly at generation 100, 200, 300, etc.
        generation = metrics.get('generation', 0)
        log_data['generation'] = generation
        
        # Log to wandb using generation as step for aligned x-axis with GA
        wandb.log(log_data, step=generation)
    
    # Checkpoint function - called at end of each task with params dict
    def checkpoint_fn(task_idx, params_dict):
        multiplier = params_dict.get('multiplier', 0.0)
        task_label = f"{task_mod}_{multiplier:.2f}".replace(".", "p")
        checkpoint_path = os.path.join(checkpoint_dir, f"task_{task_idx:02d}_{task_label}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'normalizer_params': params_dict.get('normalizer_params'),
                'policy_params': params_dict.get('policy_params'),
                'value_params': params_dict.get('value_params'),
                'task_idx': task_idx,
                'task_mod': task_mod,
                'multiplier': multiplier,
                'global_step': params_dict.get('global_step'),
                'config': {
                    'env_name': env_name,
                    'policy_hidden_sizes': policy_hidden_sizes,
                    'value_hidden_sizes': value_hidden_sizes,
                }
            }, f)
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Generation checkpoint function - called at end of each task (before switching)
    # Saves at generation 99, 199, 299, etc. for KL divergence analysis
    gen_checkpoint_dir = os.path.join(checkpoint_dir, "generations")
    os.makedirs(gen_checkpoint_dir, exist_ok=True)
    
    def generation_checkpoint_fn(generation, params_dict):
        checkpoint_path = os.path.join(gen_checkpoint_dir, f"gen_{generation:05d}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'normalizer_params': params_dict.get('normalizer_params'),
                'policy_params': params_dict.get('policy_params'),
                'value_params': params_dict.get('value_params'),
                'task_idx': params_dict.get('task_idx'),
                'task_mod': task_mod,
                'multiplier': params_dict.get('multiplier'),
                'generation': generation,
                'global_step': params_dict.get('global_step'),
                'config': {
                    'env_name': env_name,
                    'policy_hidden_sizes': policy_hidden_sizes,
                    'value_hidden_sizes': value_hidden_sizes,
                }
            }, f)
        print(f"  Generation checkpoint saved: {checkpoint_path}")
        
        # Log normalizer stats to detect observation explosions
        normalizer_params = params_dict.get('normalizer_params')
        if normalizer_params is not None:
            try:
                # Extract running statistics
                norm_mean = np.array(normalizer_params.mean)
                norm_std = np.array(normalizer_params.std)
                norm_count = normalizer_params.count if hasattr(normalizer_params, 'count') else None
                
                print(f"    Normalizer stats at gen {generation}:")
                print(f"      Count: {norm_count}")
                print(f"      Per-dim mean: {norm_mean}")
                print(f"      Per-dim std:  {norm_std}")
                
                # Check for extreme values
                extreme_threshold = 1e6
                extreme_mean_dims = np.where(np.abs(norm_mean) > extreme_threshold)[0]
                extreme_std_dims = np.where(norm_std > extreme_threshold)[0]
                
                if len(extreme_mean_dims) > 0:
                    print(f"      WARNING: Extreme mean values (>{extreme_threshold}) in dims: {extreme_mean_dims}")
                    for dim in extreme_mean_dims:
                        print(f"        Dim {dim}: mean={norm_mean[dim]:.2e}")
                        
                if len(extreme_std_dims) > 0:
                    print(f"      WARNING: Extreme std values (>{extreme_threshold}) in dims: {extreme_std_dims}")
                    for dim in extreme_std_dims:
                        print(f"        Dim {dim}: std={norm_std[dim]:.2e}")
                        
            except Exception as e:
                print(f"    Warning: Could not extract normalizer stats: {e}")
    
    # GIF callback function - saves evaluation GIFs at end of each task
    gifs_dir = os.path.join(output_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)
    
    def gif_callback_fn(task_idx, multiplier, inference_fn, env, params):
        """Save evaluation GIFs at end of task before switching."""
        try:
            num_gifs = 10  # Save 10 random trajectory GIFs per task
            max_steps = args.episode_length
            
            jit_reset = jax.jit(env.reset)
            jit_step = jax.jit(env.step)
            jit_inference_fn = jax.jit(inference_fn)
            
            task_label = f"{task_mod}_{multiplier:.2f}".replace(".", "p")
            
            # Create task-specific subdirectory
            task_gifs_dir = os.path.join(gifs_dir, f"task_{task_idx:02d}_{task_label}")
            os.makedirs(task_gifs_dir, exist_ok=True)
            
            # Collect observations for monitoring
            all_obs = []
            
            for gif_idx in range(num_gifs):
                # Use random seeds for diverse trajectories
                gif_key = jax.random.key(task_idx * 1000 + gif_idx * 37 + args.seed)
                state = jit_reset(gif_key)
                rollout = [state]
                total_reward = 0.0
                all_obs.append(np.array(state.obs))
                
                for _ in range(max_steps):
                    gif_key, act_key = jax.random.split(gif_key)
                    action, _ = jit_inference_fn(state.obs, act_key)
                    state = jit_step(state, action)
                    rollout.append(state)
                    all_obs.append(np.array(state.obs))
                    total_reward += float(state.reward)
                    if state.done:
                        break
                
                # Render every 2nd frame for smaller GIFs
                images = env.render(rollout[::2], height=240, width=320, camera="side")
                gif_path = os.path.join(task_gifs_dir, f"trajectory_{gif_idx:02d}_reward{total_reward:.0f}.gif")
                imageio.mimsave(gif_path, images, fps=30, loop=0)
            
            print(f"  Saved {num_gifs} GIFs for task {task_idx} in {task_gifs_dir}")
            
            # Log per-dimension observation statistics
            all_obs_array = np.array(all_obs)  # Shape: (num_steps, obs_dim)
            obs_mean = np.mean(all_obs_array, axis=0)
            obs_std = np.std(all_obs_array, axis=0)
            obs_min = np.min(all_obs_array, axis=0)
            obs_max = np.max(all_obs_array, axis=0)
            obs_abs_max = np.max(np.abs(all_obs_array), axis=0)
            
            print(f"  Observation stats for task {task_idx} ({task_label}):")
            print(f"    Shape: {all_obs_array.shape} (steps x obs_dim)")
            print(f"    Per-dimension mean: {obs_mean}")
            print(f"    Per-dimension std:  {obs_std}")
            print(f"    Per-dimension min:  {obs_min}")
            print(f"    Per-dimension max:  {obs_max}")
            print(f"    Per-dimension |max|: {obs_abs_max}")
            
            # Highlight any dimension with extreme values
            extreme_threshold = 1000.0
            extreme_dims = np.where(obs_abs_max > extreme_threshold)[0]
            if len(extreme_dims) > 0:
                print(f"    WARNING: Extreme values (>{extreme_threshold}) in dims: {extreme_dims}")
                for dim in extreme_dims:
                    print(f"      Dim {dim}: mean={obs_mean[dim]:.2e}, std={obs_std[dim]:.2e}, max={obs_max[dim]:.2e}, min={obs_min[dim]:.2e}")
            
            # Save observation stats to file
            obs_stats_path = os.path.join(gifs_dir, f"task_{task_idx:02d}_{task_label}_obs_stats.json")
            with open(obs_stats_path, 'w') as f:
                json.dump({
                    'task_idx': task_idx,
                    'multiplier': multiplier,
                    'task_label': task_label,
                    'obs_dim': int(all_obs_array.shape[1]),
                    'num_samples': int(all_obs_array.shape[0]),
                    'mean': obs_mean.tolist(),
                    'std': obs_std.tolist(),
                    'min': obs_min.tolist(),
                    'max': obs_max.tolist(),
                    'abs_max': obs_abs_max.tolist(),
                    'extreme_dims': extreme_dims.tolist() if len(extreme_dims) > 0 else [],
                }, f, indent=2)
                
        except Exception as e:
            print(f"  Warning: Failed to save GIFs for task {task_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Run continual training
    make_inference_fn, params, final_metrics = train_continual(
        env_factory=env_factory,
        task_multipliers=multiplier_list,
        timesteps_per_task=timesteps_per_task,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
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
        clipping_epsilon=args.clipping_epsilon,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        network_factory=network_factory,
        seed=args.seed,
        num_eval_envs=args.num_eval_envs,
        num_evals_per_task=args.num_evals_per_task,
        use_trac=args.trac,
        use_redo=args.use_redo,
        redo_frequency=args.redo_frequency,
        track_dormant=args.track_dormant,
        dormant_tau=args.dormant_tau,
        progress_fn=progress_fn,
        checkpoint_fn=checkpoint_fn,
        generation_checkpoint_fn=generation_checkpoint_fn,
        gif_callback_fn=gif_callback_fn,
    )
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "=" * 60)
    print("Continual Learning Complete!")
    print("=" * 60)
    print(f"  Environment: {env_name}")
    print(f"  Total tasks: {num_tasks}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Best reward achieved: {best_reward_overall:.2f}")
    
    # Save params
    save_path = os.path.join(output_dir, f"best_{env_name.lower()}_ppo_continual_{task_mod}_policy.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump({
            'params': params,
            'config': {
                'env_name': env_name,
                'task_mod': task_mod,
                'policy_hidden_sizes': policy_hidden_sizes,
                'value_hidden_sizes': value_hidden_sizes,
                'num_tasks': num_tasks,
                'task_multipliers': multiplier_list,
            }
        }, f)
    print(f"  Final policy saved to: {save_path}")
    
    # Final evaluation
    inference_fn = make_inference_fn(params, deterministic=True)
    key, eval_key = jax.random.split(key)
    final_multiplier = multiplier_list[-1]
    final_env = create_modified_env(env_name, task_mod, final_multiplier)
    eval_save_dir = os.path.join(output_dir, "eval_results")
    
    final_evaluation(
        final_env, inference_fn, eval_key, env_name, final_multiplier,
        num_eval_trials=100, max_steps=args.episode_length, save_dir=eval_save_dir
    )
    
    # Save training metrics to CSV
    if training_metrics_list:
        import csv
        csv_path = os.path.join(output_dir, "training_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'task', 'multiplier', 'reward', 'best_task_reward', 'best_overall_reward', 'elapsed_time'])
            writer.writeheader()
            writer.writerows(training_metrics_list)
        print(f"  Training metrics saved to: {csv_path}")
    
    wandb.finish()
    
    return best_reward_overall, params


if __name__ == "__main__":
    main()
