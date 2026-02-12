"""
Train GA on Gymnax environments (non-continual).

Supports CartPole-v1, Acrobot-v1, MountainCar-v0.
All are discrete action space environments.

Usage:
    python train_GA_gymnax.py --env CartPole-v1 --gpus 0
    python train_GA_gymnax.py --env Acrobot-v1 --gpus 0
    python train_GA_gymnax.py --env MountainCar-v0 --gpus 0
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

import jax
import jax.numpy as jnp
from jax import random, flatten_util
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import flax.linen as nn
from evosax.algorithms import SimpleGA
import gymnax
import time
import pickle
import wandb
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================================
# Policy Network (Discrete Actions)
# ============================================================================

class MLPPolicy(nn.Module):
    """MLP policy that outputs logits for discrete action selection."""
    hidden_dims: tuple = (16, 16)
    action_dim: int = 2
    
    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x  # logits


def get_flat_params(params):
    flat_params, _ = flatten_util.ravel_pytree(params)
    return flat_params


def unflatten_params(flat_params, param_template):
    _, unravel_fn = flatten_util.ravel_pytree(param_template)
    return unravel_fn(flat_params)


def create_policy_network(key, obs_dim, action_dim, hidden_dims=(16, 16)):
    policy = MLPPolicy(hidden_dims=hidden_dims, action_dim=action_dim)
    dummy_obs = jnp.zeros((obs_dim,))
    params = policy.init(key, dummy_obs)
    return policy, params


# ============================================================================
# Environment Configs
# ============================================================================

ENV_CONFIGS = {
    "CartPole-v1": {
        "num_generations": 500,
        "pop_size": 512,
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 3,
    },
    "Acrobot-v1": {
        "num_generations": 1000,
        "pop_size": 512,
        "hidden_dims": (32, 32),
        "episode_length": 500,
        "num_evals": 3,
    },
    "MountainCar-v0": {
        "num_generations": 2000,
        "pop_size": 512,
        "hidden_dims": (32, 32),
        "episode_length": 200,
        "num_evals": 3,
    },
}


# ============================================================================
# GIF Rendering
# ============================================================================

def render_cartpole_frame(obs, fig=None, ax=None):
    """Render a single CartPole frame from observation."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.clear()
    
    x, x_dot, theta, theta_dot = obs[0], obs[1], obs[2], obs[3]
    cart_width, cart_height, pole_length = 0.4, 0.2, 0.6
    
    ax.axhline(y=0, color='gray', linewidth=2)
    cart_x = float(x) - cart_width / 2
    cart = Rectangle((cart_x, 0), cart_width, cart_height, color='blue')
    ax.add_patch(cart)
    
    pole_x_end = float(x) + pole_length * np.sin(float(theta))
    pole_y_end = cart_height + pole_length * np.cos(float(theta))
    ax.plot([float(x), pole_x_end], [cart_height, pole_y_end], 'r-', linewidth=4)
    ax.plot(pole_x_end, pole_y_end, 'ro', markersize=8)
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f'x={float(x):.2f}, θ={float(theta):.2f}')
    
    return fig, ax


def render_acrobot_frame(obs, fig=None, ax=None):
    """Render Acrobot frame."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.clear()
    
    cos1, sin1, cos2, sin2 = obs[0], obs[1], obs[2], obs[3]
    link1, link2 = 1.0, 1.0
    
    p1 = (link1 * float(sin1), -link1 * float(cos1))
    p2 = (p1[0] + link2 * float(sin2), p1[1] - link2 * float(cos2))
    
    ax.plot([0, p1[0]], [0, p1[1]], 'b-', linewidth=4)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=4)
    ax.plot(0, 0, 'ko', markersize=10)
    ax.plot(p1[0], p1[1], 'bo', markersize=8)
    ax.plot(p2[0], p2[1], 'ro', markersize=8)
    ax.axhline(y=1.0, color='g', linestyle='--', linewidth=2)
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    
    return fig, ax


def render_mountaincar_frame(obs, fig=None, ax=None):
    """Render MountainCar frame."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.clear()
    
    position, velocity = float(obs[0]), float(obs[1])
    xs = np.linspace(-1.2, 0.6, 100)
    ys = np.sin(3 * xs) * 0.45 + 0.55
    
    ax.fill_between(xs, 0, ys, color='lightgreen', alpha=0.7)
    ax.plot(xs, ys, 'g-', linewidth=2)
    
    car_y = np.sin(3 * position) * 0.45 + 0.55
    ax.plot(position, car_y + 0.05, 'rs', markersize=15)
    ax.axvline(x=0.5, color='gold', linewidth=3, linestyle='--')
    
    ax.set_xlim(-1.3, 0.7)
    ax.set_ylim(0, 1.5)
    ax.set_title(f'pos={position:.2f}, vel={velocity:.3f}')
    
    return fig, ax


def save_gif(frames, path, fps=30):
    """Save frames as GIF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  [DEBUG] Saving {len(frames)} frames to {path}")
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"Saved GIF: {path}")


# ============================================================================
# Scoring Function
# ============================================================================

def make_scoring_fn(env, env_params, policy, param_template, episode_length, num_evals=1):
    """Create JIT-compiled scoring function for gymnax."""
    
    def evaluate_single(flat_params, eval_key):
        params = unflatten_params(flat_params, param_template)
        reset_key, _ = random.split(eval_key)
        obs, state = env.reset(reset_key, env_params)
        
        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
            logits = policy.apply(params, obs)
            action = jnp.argmax(logits)  # Greedy action
            
            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)
            total_reward = total_reward + reward * (1.0 - done_flag)
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
            
            return (next_obs, next_state, total_reward, done_flag, key), None
        
        (_, _, total_reward, _, _), _ = jax.lax.scan(
            step_fn, (obs, state, 0.0, 0.0, eval_key), None, length=episode_length
        )
        return total_reward
    
    vmapped_eval = jax.vmap(evaluate_single)
    
    @jax.jit
    def scoring_fn(flat_genotypes, key):
        pop_size = flat_genotypes.shape[0]
        if num_evals == 1:
            keys = random.split(key, pop_size)
            fitnesses = vmapped_eval(flat_genotypes, keys)
        else:
            all_keys = random.split(key, pop_size * num_evals)
            flat_params_repeated = jnp.repeat(flat_genotypes, num_evals, axis=0)
            all_rewards = vmapped_eval(flat_params_repeated, all_keys)
            all_rewards = all_rewards.reshape(pop_size, num_evals)
            fitnesses = jnp.mean(all_rewards, axis=1)
        return fitnesses
    
    return scoring_fn


def rollout_for_gif(env, env_params, policy, flat_params, param_template, episode_length, key, verbose=False):
    """Rollout for evaluation/GIF generation (not JIT - needs obs extraction)."""
    params = unflatten_params(flat_params, param_template)
    obs, state = env.reset(key, env_params)
    
    obs_list = [np.array(obs)]
    total_reward = 0.0
    
    for step in range(episode_length):
        logits = policy.apply(params, obs)
        action = int(jnp.argmax(logits))
        
        key, step_key = random.split(key)
        obs, state, reward, done, _ = env.step(step_key, state, action, env_params)
        obs_list.append(np.array(obs))
        total_reward += float(reward)
        
        if bool(done):
            if verbose:
                print(f"  Episode ended at step {step}, reward={total_reward:.2f}")
            break
    
    return obs_list, total_reward


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='GA on Gymnax (Non-Continual)')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'])
    parser.add_argument('--num_generations', type=int, default=None)
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--elite_ratio', type=float, default=0.5)
    parser.add_argument('--mutation_std', type=float, default=0.5)
    parser.add_argument('--num_evals', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='continual_neuroevolution_gymnax')
    parser.add_argument('--log_interval', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    
    env_name = args.env
    seed = args.seed
    trial = args.trial
    
    # Get env-specific config
    cfg = ENV_CONFIGS[env_name]
    num_generations = args.num_generations or cfg["num_generations"]
    pop_size = args.pop_size or cfg["pop_size"]
    hidden_dims = cfg["hidden_dims"]
    episode_length = cfg["episode_length"]
    num_evals = args.num_evals or cfg["num_evals"]
    
    output_dir = args.output_dir or f"projects/gymnax/ga_{env_name}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGA on {env_name} (Non-Continual)")
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}")
    print(f"  Num evals: {num_evals}")
    
    key = random.key(seed)
    
    # Create environment
    env, env_params = gymnax.make(env_name)
    
    # Get dimensions
    key, reset_key = random.split(key)
    obs, _ = env.reset(reset_key, env_params)
    obs_dim = obs.shape[-1]
    action_dim = env.action_space(env_params).n
    
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create policy
    key, init_key = random.split(key)
    policy, param_template = create_policy_network(init_key, obs_dim, action_dim, hidden_dims)
    flat_params = get_flat_params(param_template)
    num_params = flat_params.shape[0]
    print(f"  Network: {hidden_dims}, {num_params} params")
    
    # Create scoring function
    scoring_fn = make_scoring_fn(env, env_params, policy, param_template, episode_length, num_evals)
    
    # Initialize wandb
    config = {
        'env': env_name, 'num_generations': num_generations,
        'pop_size': pop_size, 'seed': seed, 'trial': trial,
        'elite_ratio': args.elite_ratio, 'mutation_std': args.mutation_std,
        'num_evals': num_evals, 'hidden_dims': hidden_dims,
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"ga_{env_name}_trial{trial}", reinit=True)
    
    # Initialize GA with multi-GPU sharding
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(np.array(devices), axis_names=('p',))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())
    parallel_sharding = NamedSharding(mesh, PartitionSpec('p'))
    
    if pop_size % num_devices != 0:
        old_pop_size = pop_size
        pop_size = (pop_size // num_devices + 1) * num_devices
        print(f"  Warning: Adjusted pop_size from {old_pop_size} to {pop_size}")
    
    std_schedule = lambda gen: args.mutation_std
    
    ga = SimpleGA(
        population_size=pop_size,
        solution=jnp.zeros(num_params),
        std_schedule=std_schedule,
    )
    ga.elite_ratio = args.elite_ratio
    ga_params = jax.device_put(ga.default_params, replicate_sharding)
    
    key, init_key = random.split(key)
    init_population = random.normal(init_key, (pop_size, num_params)) * 0.1
    init_fitness = jnp.full(pop_size, jnp.inf)
    
    key, state_init_key = random.split(key)
    ga_state = jax.jit(ga.init, out_shardings=replicate_sharding)(
        state_init_key, init_population, init_fitness, ga_params
    )
    
    # JIT compile ask/tell
    @jax.jit
    def jit_ask(key, state, params):
        population, new_state = ga.ask(key, state, params)
        population = jax.device_put(population, parallel_sharding)
        return population, new_state
    
    @jax.jit
    def jit_tell(key, population, fitness, state, params):
        return ga.tell(key, population, fitness, state, params)
    
    # Warmup JIT
    print("\nJIT compiling...")
    key, warmup_key, warmup_ask_key = random.split(key, 3)
    warmup_pop, _ = jit_ask(warmup_ask_key, ga_state, ga_params)
    _ = scoring_fn(warmup_pop, warmup_key)
    print("  JIT compilation complete!")
    
    # Training loop
    best_overall_fitness = -float('inf')
    start_time = time.time()
    
    print(f"\nStarting training...")
    
    for gen in range(num_generations):
        key, ask_key, eval_key, tell_key = random.split(key, 4)
        
        population, ga_state = jit_ask(ask_key, ga_state, ga_params)
        fitness = scoring_fn(population, eval_key)
        
        fitness_host = jax.device_get(fitness)
        population_host = jax.device_get(population)
        
        # evosax SimpleGA MINIMIZES by default, so negate fitness
        ga_state, _ = jit_tell(tell_key, population, -fitness, ga_state, ga_params)
        
        gen_best = float(np.max(fitness_host))
        gen_mean = float(np.mean(fitness_host))
        best_idx = int(np.argmax(fitness_host))
        
        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best
            best_params = population_host[best_idx].copy()
        
        wandb.log({
            "generation": gen, "best_fitness": gen_best,
            "mean_fitness": gen_mean, "best_overall": best_overall_fitness,
        })
        
        if gen % args.log_interval == 0 or gen == num_generations - 1:
            print(f"Gen {gen:4d} | Best: {gen_best:8.2f} | Mean: {gen_mean:8.2f} | Overall: {best_overall_fitness:8.2f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Best overall (training): {best_overall_fitness:.2f}")
    
    # Final evaluation with 10 trials
    print(f"\nFinal evaluation (10 trials)...")
    final_eval_rewards = []
    final_eval_num_trials = 10
    for eval_trial in range(final_eval_num_trials):
        key, eval_key = random.split(key)
        _, trial_reward = rollout_for_gif(
            env, env_params, policy, best_params, param_template, episode_length, eval_key
        )
        final_eval_rewards.append(trial_reward)
        print(f"  Trial {eval_trial + 1}: {trial_reward:.2f}")
    
    final_mean = float(np.mean(final_eval_rewards))
    final_std = float(np.std(final_eval_rewards))
    final_max = float(np.max(final_eval_rewards))
    final_min = float(np.min(final_eval_rewards))
    
    print(f"\nFinal evaluation results:")
    print(f"  Mean: {final_mean:.2f} +/- {final_std:.2f}")
    print(f"  Min: {final_min:.2f}, Max: {final_max:.2f}")
    print(f"  Training best: {best_overall_fitness:.2f}")
    
    wandb.log({
        "final_eval_mean": final_mean,
        "final_eval_std": final_std,
        "final_eval_max": final_max,
        "final_eval_min": final_min,
    })
    
    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"ga_{env_name}_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': np.array(best_params),
            'param_template': param_template,
            'best_fitness': best_overall_fitness,
            'final_eval_mean': final_mean,
            'final_eval_std': final_std,
            'config': config,
        }, f)
    print(f"Saved: {ckpt_path}")
    
    # Generate 10 GIFs (one per evaluation trial)
    try:
        gifs_dir = os.path.join(output_dir, "gifs")
        os.makedirs(gifs_dir, exist_ok=True)
        
        render_fn = {
            "CartPole-v1": render_cartpole_frame,
            "Acrobot-v1": render_acrobot_frame,
            "MountainCar-v0": render_mountaincar_frame,
        }[env_name]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        for gif_idx in range(10):
            key, gif_key = random.split(key)
            obs_list, reward = rollout_for_gif(
                env, env_params, policy, best_params, param_template, episode_length, gif_key
            )
            
            frames = []
            obs_to_render = obs_list[:min(len(obs_list), 200)]
            for obs in obs_to_render:
                render_fn(obs, fig, ax)
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
                frames.append(frame)
            
            gif_path = os.path.join(gifs_dir, f"eval_{gif_idx:02d}_reward{reward:.0f}.gif")
            save_gif(frames, gif_path, fps=30)
        
        plt.close(fig)
        print(f"Saved 10 GIFs to: {gifs_dir}")
        
    except Exception as e:
        print(f"Warning: GIF generation failed: {e}")
    
    wandb.finish()
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
