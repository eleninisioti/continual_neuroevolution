"""
Train DNS (Dominated Novelty Search) on Gymnax environments (CONTINUAL).

Task changes every 200 generations by adding observation noise.
Noise vector = normal(obs_size) * noise_range

Custom DNS implementation matching the mujoco version.

Supports CartPole-v1, Acrobot-v1, MountainCar-v0.

Usage:
    python train_DNS_gymnax_continual.py --env CartPole-v1 --gpus 0
    python train_DNS_gymnax_continual.py --env Acrobot-v1 --gpus 0
"""

import argparse
import os
import sys
import time

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
import flax.linen as nn
import gymnax
import pickle
import wandb
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================================
# Environment Configs
# ============================================================================

ENV_CONFIGS = {
    "CartPole-v1": {
        "num_generations": 1000,  # 5 tasks x 200 gens
        "pop_size": 512,
        "batch_size": 256,
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 3,
        "k": 3,
        "iso_sigma": 0.05,
        "line_sigma": 0.5,
        "task_interval": 200,
        "num_tasks": 5,
    },
    "Acrobot-v1": {
        "num_generations": 1000,
        "pop_size": 512,
        "batch_size": 256,
        "hidden_dims": (32, 32),
        "episode_length": 500,
        "num_evals": 3,
        "k": 3,
        "iso_sigma": 0.05,
        "line_sigma": 0.5,
        "task_interval": 200,
        "num_tasks": 5,
    },
    "MountainCar-v0": {
        "num_generations": 1000,
        "pop_size": 512,
        "batch_size": 256,
        "hidden_dims": (32, 32),
        "episode_length": 200,
        "num_evals": 3,
        "k": 3,
        "iso_sigma": 0.05,
        "line_sigma": 0.5,
        "task_interval": 200,
        "num_tasks": 5,
    },
}


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
# DNS Algorithm
# ============================================================================

def compute_novelty(descriptors: jnp.ndarray, k: int = 3) -> jnp.ndarray:
    """Compute novelty scores as average distance to k nearest neighbors."""
    descriptors = jnp.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
    
    diffs = descriptors[:, None, :] - descriptors[None, :, :]
    distances = jnp.linalg.norm(diffs, axis=-1)
    distances = distances + jnp.eye(distances.shape[0]) * 1e10
    
    k_nearest_distances = jnp.sort(distances, axis=1)[:, :k]
    novelty = jnp.mean(k_nearest_distances, axis=1)
    novelty = jnp.nan_to_num(novelty, nan=0.0, posinf=0.0, neginf=0.0)
    
    return novelty


def dns_selection(
    genotypes, fitnesses, descriptors,
    new_genotypes, new_fitnesses, new_descriptors,
    population_size, k,
):
    """DNS selection: combine parent and offspring, select based on combined fitness+novelty score."""
    combined_genotypes = jnp.concatenate([genotypes, new_genotypes], axis=0)
    combined_fitnesses = jnp.concatenate([fitnesses, new_fitnesses], axis=0)
    combined_descriptors = jnp.concatenate([descriptors, new_descriptors], axis=0)
    
    combined_novelties = compute_novelty(combined_descriptors, k)
    combined_fitnesses = jnp.nan_to_num(combined_fitnesses, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Normalize fitness to [0, 1]
    f_min, f_max = jnp.min(combined_fitnesses), jnp.max(combined_fitnesses)
    f_range = jnp.maximum(f_max - f_min, 1e-8)
    norm_fitness = (combined_fitnesses - f_min) / f_range
    
    # Normalize novelty to [0, 1]
    n_min, n_max = jnp.min(combined_novelties), jnp.max(combined_novelties)
    n_range = jnp.maximum(n_max - n_min, 1e-8)
    norm_novelty = (combined_novelties - n_min) / n_range
    
    combined_scores = jnp.nan_to_num(norm_fitness + norm_novelty, nan=-1e6)
    selected_indices = jnp.argsort(combined_scores)[-population_size:]
    
    return (
        combined_genotypes[selected_indices],
        combined_fitnesses[selected_indices],
        combined_descriptors[selected_indices],
        combined_novelties[selected_indices],
    )


def isoline_variation(genotypes, key, iso_sigma=0.05, line_sigma=0.5, batch_size=256):
    """Isoline variation operator for generating offspring."""
    pop_size = genotypes.shape[0]
    param_dim = genotypes.shape[1]
    
    key, k1, k2, k3, k4 = random.split(key, 5)
    parent1_indices = random.randint(k1, (batch_size,), 0, pop_size)
    parent2_indices = random.randint(k2, (batch_size,), 0, pop_size)
    
    parent1 = genotypes[parent1_indices]
    parent2 = genotypes[parent2_indices]
    
    alpha = random.uniform(k3, (batch_size, param_dim), minval=-line_sigma, maxval=1.0 + line_sigma)
    offspring = parent1 + alpha * (parent2 - parent1)
    
    noise = random.normal(k4, (batch_size, param_dim)) * iso_sigma
    offspring = offspring + noise
    
    return offspring


# ============================================================================
# GIF Rendering
# ============================================================================

def render_cartpole_frame(obs, fig=None, ax=None):
    """Render a single CartPole frame from observation."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    ax.clear()
    cart_x = float(obs[0])
    pole_angle = float(obs[2])
    
    cart_width = 0.4
    cart_height = 0.2
    pole_length = 0.6
    
    cart_left = cart_x - cart_width / 2
    cart_bottom = 0
    ax.add_patch(Rectangle((cart_left, cart_bottom), cart_width, cart_height, 
                           facecolor='blue', edgecolor='black'))
    
    pole_x = cart_x + pole_length * np.sin(pole_angle)
    pole_y = cart_height + pole_length * np.cos(pole_angle)
    ax.plot([cart_x, pole_x], [cart_height, pole_y], 'r-', linewidth=3)
    ax.plot(pole_x, pole_y, 'ro', markersize=8)
    
    ax.axhline(y=0, color='black', linewidth=2)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig, ax


def render_mountaincar_frame(obs, fig=None, ax=None):
    """Render a single MountainCar frame from observation."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
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


def render_acrobot_frame(obs, fig=None, ax=None):
    """Render a single Acrobot frame from observation."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    ax.clear()
    
    cos_t1, sin_t1 = float(obs[0]), float(obs[1])
    cos_t2, sin_t2 = float(obs[2]), float(obs[3])
    
    l1, l2 = 1.0, 1.0
    
    x1 = l1 * sin_t1
    y1 = -l1 * cos_t1
    
    x2 = x1 + l2 * sin_t2
    y2 = y1 - l2 * cos_t2
    
    ax.plot([0, x1], [0, y1], 'b-', linewidth=4)
    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=4)
    
    ax.plot(0, 0, 'ko', markersize=10)
    ax.plot(x1, y1, 'bo', markersize=8)
    ax.plot(x2, y2, 'ro', markersize=8)
    
    ax.axhline(y=l1, color='g', linestyle='--', alpha=0.5)
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
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

def make_scoring_fn(env, env_params, policy, param_template, episode_length, env_name, num_evals=1):
    """Create scoring function that returns fitness and behavioral descriptor with noise."""
    
    def evaluate_single(flat_params, eval_key, noise_vector):
        """Evaluate a single individual and return fitness + descriptor."""
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
            
            return (next_obs, next_state, total_reward, done_flag, key), obs
        
        key = eval_key
        (final_obs, _, total_reward, _, _), all_obs = jax.lax.scan(
            step_fn, (obs, state, 0.0, 0.0, key), None, length=episode_length
        )
        
        # Extract behavioral descriptor based on environment
        if 'CartPole' in env_name:
            descriptor = jnp.array([final_obs[0], final_obs[2]])  # cart pos, pole angle
        elif 'MountainCar' in env_name:
            descriptor = final_obs  # position and velocity
        elif 'Acrobot' in env_name:
            descriptor = final_obs[:2]  # joint angles (cos, sin of first joint)
        else:
            descriptor = final_obs[:2]
        
        return total_reward, descriptor
    
    vmapped_eval = jax.vmap(evaluate_single, in_axes=(0, 0, None))
    
    @jax.jit
    def scoring_fn(flat_genotypes, key, noise_vector):
        pop_size = flat_genotypes.shape[0]
        if num_evals == 1:
            keys = random.split(key, pop_size)
            fitnesses, descriptors = vmapped_eval(flat_genotypes, keys, noise_vector)
        else:
            all_keys = random.split(key, pop_size * num_evals)
            flat_params_repeated = jnp.repeat(flat_genotypes, num_evals, axis=0)
            all_fitnesses, all_descriptors = vmapped_eval(flat_params_repeated, all_keys, noise_vector)
            all_fitnesses = all_fitnesses.reshape(pop_size, num_evals)
            all_descriptors = all_descriptors.reshape(pop_size, num_evals, -1)
            fitnesses = jnp.mean(all_fitnesses, axis=1)
            descriptors = jnp.mean(all_descriptors, axis=1)
        return fitnesses, descriptors
    
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
# Diversity Metrics
# ============================================================================

def compute_fitness_diversity(fitnesses):
    return float(jnp.std(fitnesses))


def compute_genomic_diversity(genotypes, sample_size=32):
    """Compute genomic diversity using sampling to avoid OOM."""
    pop_size = genotypes.shape[0]
    if pop_size < 2:
        return 0.0
    
    actual_sample = min(sample_size, pop_size)
    indices = np.random.choice(pop_size, size=actual_sample, replace=False)
    sampled = genotypes[indices]
    
    sampled_np = np.array(sampled)
    g_min = np.min(sampled_np, axis=0, keepdims=True)
    g_max = np.max(sampled_np, axis=0, keepdims=True)
    g_range = np.maximum(g_max - g_min, 1e-8)
    norm_genotypes = (sampled_np - g_min) / g_range
    
    diffs = norm_genotypes[:, None, :] - norm_genotypes[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    
    mask = np.triu(np.ones((actual_sample, actual_sample)), k=1)
    mean_dist = np.sum(distances * mask) / np.sum(mask)
    
    return float(mean_dist)


# ============================================================================
# Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='DNS on Gymnax (Continual)')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'])
    parser.add_argument('--num_generations', type=int, default=None)
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--iso_sigma', type=float, default=None)
    parser.add_argument('--line_sigma', type=float, default=None)
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
    batch_size = args.batch_size or cfg["batch_size"]
    hidden_dims = cfg["hidden_dims"]
    episode_length = cfg["episode_length"]
    num_evals = args.num_evals or cfg["num_evals"]
    k = args.k or cfg["k"]
    iso_sigma = args.iso_sigma or cfg["iso_sigma"]
    line_sigma = args.line_sigma or cfg["line_sigma"]
    task_interval = args.task_interval
    noise_range = args.noise_range
    
    output_dir = args.output_dir or f"projects/gymnax/dns_{env_name}_continual/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    gifs_dir = os.path.join(output_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"DNS on {env_name} (CONTINUAL)")
    print("=" * 60)
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}, Batch: {batch_size}")
    print(f"  Task interval: {task_interval} gens")
    print(f"  Noise range: {noise_range}")
    print(f"  k (novelty neighbors): {k}")
    print(f"  iso_sigma: {iso_sigma}, line_sigma: {line_sigma}")
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
    scoring_fn = make_scoring_fn(env, env_params, policy, param_template, episode_length, env_name, num_evals)
    
    # Initialize wandb
    config = {
        'env': env_name, 'num_generations': num_generations,
        'pop_size': pop_size, 'batch_size': batch_size, 'k': k,
        'iso_sigma': iso_sigma, 'line_sigma': line_sigma,
        'seed': seed, 'trial': trial, 'num_evals': num_evals,
        'hidden_dims': hidden_dims, 'task_interval': task_interval,
        'noise_range': noise_range, 'continual': True,
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"dns_{env_name}_continual_trial{trial}", reinit=True)
    
    # Initialize population
    key, pop_key = jax.random.split(key)
    population = random.normal(pop_key, (pop_size, num_params)) * 0.1
    
    # Generate initial noise vector (task 0)
    key, noise_key = random.split(key)
    noise_vector = random.normal(noise_key, (obs_dim,)) * noise_range
    current_task = 0
    
    # Initial evaluation
    print(f"\nEvaluating initial population...")
    key, eval_key = jax.random.split(key)
    fitnesses, descriptors = scoring_fn(population, eval_key, noise_vector)
    novelties = compute_novelty(descriptors, k)
    
    print(f"  Initial best fitness: {float(jnp.max(fitnesses)):.2f}")
    print(f"  Task 0 noise vector: {jax.device_get(noise_vector)}")
    print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")
    
    # Training loop
    best_overall_fitness = -float('inf')
    best_params = None
    start_time = time.time()
    render_fn = get_render_fn(env_name)
    
    print(f"\nStarting continual training...")
    
    for gen in range(num_generations):
        # Check if we need to switch task
        if gen > 0 and gen % task_interval == 0:
            # Save GIF of best solution BEFORE switching task
            if best_params is not None:
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
            
            # Re-evaluate population with new noise
            key, eval_key = jax.random.split(key)
            fitnesses, descriptors = scoring_fn(population, eval_key, noise_vector)
            novelties = compute_novelty(descriptors, k)
        
        # Generate offspring
        key, var_key = jax.random.split(key)
        offspring = isoline_variation(population, var_key, iso_sigma, line_sigma, batch_size)
        
        # Evaluate offspring
        key, eval_key = jax.random.split(key)
        offspring_fitnesses, offspring_descriptors = scoring_fn(offspring, eval_key, noise_vector)
        
        # DNS selection
        population, fitnesses, descriptors, novelties = dns_selection(
            population, fitnesses, descriptors,
            offspring, offspring_fitnesses, offspring_descriptors,
            pop_size, k
        )
        
        # Copy to host for numpy operations
        fitness_host = jax.device_get(fitnesses)
        population_host = jax.device_get(population)
        
        gen_best = float(np.max(fitness_host))
        gen_mean = float(np.mean(fitness_host))
        best_idx = int(np.argmax(fitness_host))
        
        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best
            best_params = population_host[best_idx].copy()
        
        # Diversity metrics
        fitness_div = compute_fitness_diversity(fitnesses)
        genomic_div = compute_genomic_diversity(population)
        mean_novelty = float(jnp.nanmean(novelties))
        
        wandb.log({
            "generation": gen, "best_fitness": gen_best,
            "mean_fitness": gen_mean, "best_overall": best_overall_fitness,
            "fitness_diversity": fitness_div, "genomic_diversity": genomic_div,
            "mean_novelty": mean_novelty, "task": current_task,
            "noise_magnitude": float(jnp.linalg.norm(noise_vector)),
        })
        
        if gen % args.log_interval == 0 or gen == num_generations - 1:
            print(f"Gen {gen:4d} | Task {current_task} | Best: {gen_best:8.2f} | Mean: {gen_mean:8.2f} | Overall: {best_overall_fitness:8.2f} | Nov: {mean_novelty:.3f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Best overall: {best_overall_fitness:.2f}")
    print(f"  Total tasks: {current_task + 1}")
    
    # Save final GIF
    if best_params is not None:
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
    ckpt_path = os.path.join(output_dir, f"dns_{env_name}_continual_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': np.array(best_params) if best_params is not None else None,
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
