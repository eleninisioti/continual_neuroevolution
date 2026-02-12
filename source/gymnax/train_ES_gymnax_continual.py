"""
Train OpenES on Gymnax environments (CONTINUAL).

Task changes every 200 generations by adding observation noise.
Noise vector = normal(obs_size) * noise_range

Supports CartPole-v1, Acrobot-v1, MountainCar-v0.

Usage:
    python train_ES_gymnax_continual.py --env CartPole-v1 --gpus 0
    python train_ES_gymnax_continual.py --env Acrobot-v1 --gpus 0
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
import optax
from evosax.algorithms import Open_ES
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
        "num_generations": 1000,  # 5 tasks x 200 gens
        "pop_size": 512,
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 3,
        "sigma": 0.5,
        "learning_rate": 0.01,
        "task_interval": 200,
        "num_tasks": 5,
    },
    "Acrobot-v1": {
        "num_generations": 1000,
        "pop_size": 512,
        "hidden_dims": (32, 32),
        "episode_length": 500,
        "num_evals": 3,
        "sigma": 0.5,
        "learning_rate": 0.01,
        "task_interval": 200,
        "num_tasks": 5,
    },
    "MountainCar-v0": {
        "num_generations": 1000,
        "pop_size": 512,
        "hidden_dims": (32, 32),
        "episode_length": 200,
        "num_evals": 3,
        "sigma": 0.5,
        "learning_rate": 0.01,
        "task_interval": 200,
        "num_tasks": 5,
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
    l1, l2 = 1.0, 1.0
    
    x1 = l1 * float(sin1)
    y1 = -l1 * float(cos1)
    x2 = x1 + l2 * float(sin2)
    y2 = y1 - l2 * float(cos2)
    
    ax.plot([0, x1], [0, y1], 'b-', linewidth=4)
    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=4)
    ax.plot(0, 0, 'ko', markersize=10)
    ax.plot(x1, y1, 'bo', markersize=8)
    ax.plot(x2, y2, 'ro', markersize=8)
    
    ax.axhline(y=l1, color='g', linestyle='--', alpha=0.5)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    
    return fig, ax


def render_mountaincar_frame(obs, fig=None, ax=None):
    """Render MountainCar frame."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.clear()
    
    pos = float(obs[0])
    x = np.linspace(-1.2, 0.6, 100)
    y = np.sin(3 * x) * 0.45 + 0.55
    ax.plot(x, y, 'k-', linewidth=2)
    ax.fill_between(x, 0, y, alpha=0.3)
    
    car_y = np.sin(3 * pos) * 0.45 + 0.55
    ax.plot(pos, car_y + 0.05, 'ro', markersize=15)
    ax.plot(0.5, np.sin(3 * 0.5) * 0.45 + 0.55, 'g^', markersize=15)
    
    ax.set_xlim(-1.3, 0.7)
    ax.set_ylim(0, 1.2)
    ax.axis('off')
    
    return fig, ax


def get_render_fn(env_name):
    if 'CartPole' in env_name:
        return render_cartpole_frame
    elif 'Acrobot' in env_name:
        return render_acrobot_frame
    elif 'MountainCar' in env_name:
        return render_mountaincar_frame
    return render_cartpole_frame


# ============================================================================
# Scoring Function with Observation Noise
# ============================================================================

def make_scoring_fn(env, env_params, policy, param_template, episode_length, num_evals=1):
    """Create scoring function that accepts noise_vector for continual learning."""
    
    def evaluate_single(flat_params, eval_key, noise_vector):
        params = unflatten_params(flat_params, param_template)
        obs, state = env.reset(eval_key, env_params)
        
        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
            # Add noise to observation for continual learning
            noisy_obs = obs + noise_vector
            logits = policy.apply(params, noisy_obs)
            action = jnp.argmax(logits)  # Deterministic for evaluation
            
            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)
            
            total_reward = total_reward + reward * (1.0 - done_flag)
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
            
            return (next_obs, next_state, total_reward, done_flag, key), None
        
        key = eval_key
        (_, _, total_reward, _, _), _ = jax.lax.scan(
            step_fn, (obs, state, 0.0, 0.0, key), None, length=episode_length
        )
        return total_reward
    
    vmapped_eval = jax.vmap(evaluate_single, in_axes=(0, 0, None))
    
    @jax.jit
    def scoring_fn(flat_genotypes, key, noise_vector):
        pop_size = flat_genotypes.shape[0]
        if num_evals == 1:
            keys = random.split(key, pop_size)
            fitnesses = vmapped_eval(flat_genotypes, keys, noise_vector)
        else:
            all_keys = random.split(key, pop_size * num_evals)
            flat_params_repeated = jnp.repeat(flat_genotypes, num_evals, axis=0)
            all_rewards = vmapped_eval(flat_params_repeated, all_keys, noise_vector)
            all_rewards = all_rewards.reshape(pop_size, num_evals)
            fitnesses = jnp.mean(all_rewards, axis=1)
        return fitnesses
    
    return scoring_fn


def rollout_for_gif(env, env_params, policy, flat_params, param_template, episode_length, key, noise_vector):
    """Rollout for GIF generation with noise."""
    params = unflatten_params(flat_params, param_template)
    obs, state = env.reset(key, env_params)
    
    obs_list = [np.array(obs)]
    total_reward = 0.0
    
    for _ in range(episode_length):
        noisy_obs = obs + noise_vector
        logits = policy.apply(params, noisy_obs)
        action = int(jnp.argmax(logits))
        
        key, step_key = random.split(key)
        obs, state, reward, done, _ = env.step(step_key, state, action, env_params)
        total_reward += float(reward)
        obs_list.append(np.array(obs))
        
        if bool(done):
            break
    
    return obs_list, total_reward


# ============================================================================
# Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='OpenES on Gymnax (Continual)')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'])
    parser.add_argument('--num_generations', type=int, default=None)
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_evals', type=int, default=None)
    parser.add_argument('--task_interval', type=int, default=200)
    parser.add_argument('--noise_range', type=float, default=1.0,
                        help='Scale for observation noise (task definition)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='continual_neuroevolution_gymnax')
    parser.add_argument('--log_interval', type=int, default=10)
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

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
    sigma = args.sigma or cfg["sigma"]
    learning_rate = args.learning_rate or cfg["learning_rate"]
    task_interval = args.task_interval
    noise_range = args.noise_range
    
    output_dir = args.output_dir or f"projects/gymnax/es_{env_name}_continual/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    gifs_dir = os.path.join(output_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"OpenES on {env_name} (CONTINUAL)")
    print("=" * 60)
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}")
    print(f"  Task interval: {task_interval} gens")
    print(f"  Noise range: {noise_range}")
    print(f"  Sigma: {sigma}, LR: {learning_rate}")
    print(f"  Num evals: {num_evals}")
    
    key = jax.random.key(seed)
    
    # Create environment
    env, env_params = gymnax.make(env_name)
    env_params = env_params.replace(max_steps_in_episode=episode_length)
    
    # Get dimensions
    key, reset_key = jax.random.split(key)
    obs, _ = env.reset(reset_key, env_params)
    obs_dim = obs.shape[-1]
    action_dim = env.action_space(env_params).n
    
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create policy
    key, init_key = jax.random.split(key)
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
        'sigma': sigma, 'learning_rate': learning_rate,
        'num_evals': num_evals, 'hidden_dims': hidden_dims,
        'task_interval': task_interval, 'noise_range': noise_range,
        'continual': True,
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"es_{env_name}_continual_trial{trial}", reinit=True)
    
    # Setup ES with multi-GPU sharding
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(np.array(devices), axis_names=('p',))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())
    parallel_sharding = NamedSharding(mesh, PartitionSpec('p'))
    
    if pop_size % num_devices != 0:
        old_pop_size = pop_size
        pop_size = (pop_size // num_devices + 1) * num_devices
        print(f"  Warning: Adjusted pop_size from {old_pop_size} to {pop_size}")
    
    # Custom std schedule (constant)
    std_schedule = lambda gen: sigma
    
    # OpenES uses Adam optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    
    es = Open_ES(
        population_size=pop_size,
        solution=jnp.zeros(num_params),
        std_schedule=std_schedule,
        optimizer=optimizer,
        use_antithetic_sampling=True,
    )
    es_params = jax.device_put(es.default_params, replicate_sharding)
    
    key, init_key = random.split(key)
    es_state = es.init(init_key, jnp.zeros(num_params), es_params)
    es_state = jax.device_put(es_state, replicate_sharding)
    
    @jax.jit
    def jit_ask(key, state, params):
        population, new_state = es.ask(key, state, params)
        population = jax.device_put(population, parallel_sharding)
        return population, new_state
    
    @jax.jit
    def jit_tell(key, population, fitness, state, params):
        return es.tell(key, population, fitness, state, params)
    
    # Generate initial noise vector (task 0)
    key, noise_key = random.split(key)
    noise_vector = random.normal(noise_key, (obs_dim,)) * noise_range
    current_task = 0
    
    print(f"\n  Task 0 noise vector: {jax.device_get(noise_vector)}")
    print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")
    
    # Warmup JIT
    print("\nJIT compiling...")
    key, warmup_key, warmup_ask_key = random.split(key, 3)
    warmup_pop, _ = jit_ask(warmup_ask_key, es_state, es_params)
    _ = scoring_fn(warmup_pop, warmup_key, noise_vector)
    print("  JIT compilation complete!")
    
    # Training loop
    best_overall_fitness = -float('inf')
    start_time = time.time()
    render_fn = get_render_fn(env_name)
    
    print(f"\nStarting continual training...")
    
    for gen in range(num_generations):
        # Check if we need to switch task
        if gen > 0 and gen % task_interval == 0:
            # Save GIF of best solution (es_state.mean) BEFORE switching task
            best_mean_params = jax.device_get(es_state.mean)
            try:
                key, gif_key = random.split(key)
                obs_list, total_reward = rollout_for_gif(
                    env, env_params, policy, best_mean_params, param_template, 
                    episode_length, gif_key, noise_vector
                )
                
                frames = []
                fig, ax = None, None
                for obs in obs_list[::2]:
                    fig, ax = render_fn(obs, fig, ax)
                    fig.canvas.draw()
                    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
                    frames.append(image)
                plt.close(fig)
                
                gif_path = os.path.join(gifs_dir, f"task{current_task}_gen{gen}_reward{total_reward:.0f}.gif")
                imageio.mimsave(gif_path, frames, fps=30)
                print(f"  Saved GIF: {gif_path} ({len(frames)} frames)")
            except Exception as e:
                print(f"  Warning: Failed to save GIF: {e}")
            
            # Switch to new task
            current_task += 1
            key, noise_key = random.split(key)
            noise_vector = random.normal(noise_key, (obs_dim,)) * noise_range
            print(f"\n>>> Task {current_task} started at gen {gen}")
            print(f"  Full noise: {jax.device_get(noise_vector)}")
            print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")
        
        key, ask_key, eval_key, tell_key = random.split(key, 4)
        
        population, es_state = jit_ask(ask_key, es_state, es_params)
        fitness = scoring_fn(population, eval_key, noise_vector)
        
        fitness_host = jax.device_get(fitness)
        
        # evosax minimizes, so negate fitness for maximization
        es_state, _ = jit_tell(tell_key, population, -fitness, es_state, es_params)
        
        gen_best = float(np.max(fitness_host))
        gen_mean = float(np.mean(fitness_host))
        
        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best
        
        wandb.log({
            "generation": gen, "best_fitness": gen_best,
            "mean_fitness": gen_mean, "best_overall": best_overall_fitness,
            "task": current_task,
            "noise_magnitude": float(jnp.linalg.norm(noise_vector)),
        })
        
        if gen % args.log_interval == 0 or gen == num_generations - 1:
            print(f"Gen {gen:4d} | Task {current_task} | Best: {gen_best:8.2f} | Mean: {gen_mean:8.2f} | Overall: {best_overall_fitness:8.2f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Best overall: {best_overall_fitness:.2f}")
    print(f"  Total tasks: {current_task + 1}")
    
    # Use ES mean as the best solution
    best_params = jax.device_get(es_state.mean)
    
    # Save final GIF
    try:
        key, gif_key = random.split(key)
        obs_list, total_reward = rollout_for_gif(
            env, env_params, policy, best_params, param_template, 
            episode_length, gif_key, noise_vector
        )
        
        frames = []
        fig, ax = None, None
        for obs in obs_list[::2]:
            fig, ax = render_fn(obs, fig, ax)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
            frames.append(image)
        plt.close(fig)
        
        gif_path = os.path.join(gifs_dir, f"task{current_task}_final_reward{total_reward:.0f}.gif")
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Saved final GIF: {gif_path} ({len(frames)} frames)")
    except Exception as e:
        print(f"Warning: Failed to save final GIF: {e}")
    
    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"es_{env_name}_continual_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': np.array(best_params),
            'param_template': param_template,
            'best_fitness': best_overall_fitness,
            'config': config,
            'final_task': current_task,
        }, f)
    print(f"Saved: {ckpt_path}")
    
    wandb.finish()
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
