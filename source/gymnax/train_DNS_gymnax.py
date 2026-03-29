"""
Train DNS (Dominated Novelty Search) on Gymnax environments (non-continual).

Custom DNS implementation matching the mujoco version.

Supports CartPole-v1, Acrobot-v1, MountainCar-v0.
All are discrete action space environments.

Usage:
    python train_DNS_gymnax.py --env CartPole-v1 --gpus 0
    python train_DNS_gymnax.py --env Acrobot-v1 --gpus 0
    python train_DNS_gymnax.py --env MountainCar-v0 --gpus 0
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
# Logging Helper
# ============================================================================

class Tee:
    """Duplicate stdout to a file and console."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w', buffering=1)
        self.stdout = sys.stdout
    
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        self.file.close()


# ============================================================================
# Environment Configs
# ============================================================================

ENV_CONFIGS = {
    "CartPole-v1": {
        "num_generations": 500,
        "pop_size": 512,
        "batch_size": 256,
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 10,
        "k": 3,
        "iso_sigma": 0.05,
        "line_sigma": 0.5,
    },
    "Acrobot-v1": {
        "num_generations": 1000,
        "pop_size": 512,
        "batch_size": 256,
        "hidden_dims": (32, 32),
        "episode_length": 500,
        "num_evals": 10,
        "k": 3,
        "iso_sigma": 0.05,
        "line_sigma": 0.5,
    },
    "MountainCar-v0": {
        "num_generations": 2000,
        "pop_size": 512,
        "batch_size": 256,
        "hidden_dims": (32, 32),
        "episode_length": 500,
        "num_evals": 10,
        "k": 3,
        "iso_sigma": 0.05,
        "line_sigma": 0.5,
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

def _compute_dominated_novelty(fitness, descriptor, k):
    """Compute dominated novelty — exact port of QDax dns_repertoire.py.
    
    For each individual, dominated novelty is the mean distance in descriptor
    space to the k nearest neighbors that have fitness >= its own.
    Returns NaN for the fittest individuals (no fitter neighbors exist).
    NaN sorts highest in descending argsort, so they are always kept.
    """
    n = fitness.shape[0]
    if n <= 1:
        return jnp.full(n, jnp.nan)
    k = min(k, n - 1)
    valid = fitness != -jnp.inf

    # Neighbor mask excluding self
    neighbor = valid[:, None] & valid[None, :]
    neighbor = neighbor & ~jnp.eye(n, dtype=bool)

    # Fitter-or-equal mask: fitter[i,j] = True if fitness[j] >= fitness[i]
    fitter = fitness[:, None] <= fitness[None, :]
    fitter = jnp.where(neighbor, fitter, False)

    # Pairwise distances
    distance = jnp.linalg.norm(
        descriptor[:, None, :] - descriptor[None, :, :], axis=-1
    )
    distance = jnp.where(neighbor, distance, jnp.inf)

    # Distances to fitter neighbors only
    distance_fitter = jnp.where(fitter, distance, jnp.inf)

    # Dominated novelty: mean distance to k nearest fitter neighbors
    values_fit, indices_fit = jax.vmap(
        lambda x: jax.lax.top_k(-x, k)
    )(distance_fitter)
    dominated_novelty = jnp.mean(
        -values_fit,
        axis=-1,
        where=jnp.take_along_axis(fitter, indices_fit, axis=-1),
    )

    return dominated_novelty


def dns_selection(
    genotypes, fitnesses, descriptors,
    new_genotypes, new_fitnesses, new_descriptors,
    population_size, k,
):
    """DNS selection — follows QDax DominatedNoveltyRepertoire.add().
    
    Combines parents and offspring, computes dominated novelty on the combined
    pool, and keeps the top population_size individuals by dominated novelty.
    """
    # Combine candidates
    combined_genotypes = jnp.concatenate([genotypes, new_genotypes], axis=0)
    combined_fitnesses = jnp.concatenate([fitnesses, new_fitnesses], axis=0)
    combined_descriptors = jnp.concatenate([descriptors, new_descriptors], axis=0)

    # Compute dominated novelty on combined pool
    dominated_novelty = _compute_dominated_novelty(
        combined_fitnesses, combined_descriptors, k
    )

    # Use dominated novelty as meta-fitness, invalid individuals get -inf
    valid = combined_fitnesses != -jnp.inf
    meta_fitness = jnp.where(valid, dominated_novelty, -jnp.inf)

    # Select survivors (NaN from fittest individuals sorts highest in descending order)
    indices = jnp.argsort(meta_fitness)[::-1]
    survivor_indices = indices[:population_size]

    return (
        combined_genotypes[survivor_indices],
        combined_fitnesses[survivor_indices],
        combined_descriptors[survivor_indices],
        dominated_novelty[survivor_indices],
    )


def isoline_variation(genotypes, key, iso_sigma=0.05, line_sigma=0.5, batch_size=256):
    """Iso+Line-DD variation operator — exact port of QDax mutation_operators.py."""
    pop_size = genotypes.shape[0]
    param_dim = genotypes.shape[1]
    
    key, k1, k2, k3, k4 = random.split(key, 5)
    parent1_indices = random.randint(k1, (batch_size,), 0, pop_size)
    parent2_indices = random.randint(k2, (batch_size,), 0, pop_size)
    
    x1 = genotypes[parent1_indices]
    x2 = genotypes[parent2_indices]
    
    # QDax: line_noise is a scalar per individual from normal distribution
    line_noise = random.normal(k3, (batch_size,)) * line_sigma
    # QDax: iso_noise is per-parameter from normal distribution
    iso_noise = random.normal(k4, (batch_size, param_dim)) * iso_sigma
    
    # QDax formula: offspring = (x1 + iso_noise) + (x2 - x1) * line_noise
    offspring = (x1 + iso_noise) + (x2 - x1) * line_noise[:, None]
    
    return offspring


# ============================================================================
# GIF Rendering
# ============================================================================

def render_cartpole_frame(obs, fig=None, ax=None, step=None):
    """Render a single CartPole frame from observation."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.clear()
    
    x, x_dot, theta, theta_dot = obs[0], obs[1], obs[2], obs[3]
    
    # Cart dimensions
    cart_width = 0.4
    cart_height = 0.2
    pole_length = 0.6
    
    # Draw track
    ax.axhline(y=0, color='gray', linewidth=2)
    
    # Draw cart
    cart_x = float(x) - cart_width / 2
    cart = Rectangle((cart_x, 0), cart_width, cart_height, color='blue')
    ax.add_patch(cart)
    
    # Draw pole
    pole_x_end = float(x) + pole_length * np.sin(float(theta))
    pole_y_end = cart_height + pole_length * np.cos(float(theta))
    ax.plot([float(x), pole_x_end], [cart_height, pole_y_end], 'r-', linewidth=4)
    
    # Draw pole tip
    ax.plot(pole_x_end, pole_y_end, 'ro', markersize=8)
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f'CartPole - Step {step}' if step is not None else 'CartPole')
    
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    
    return image, fig, ax


def render_mountaincar_frame(obs, fig=None, ax=None, step=None):
    """Render a single MountainCar frame from observation."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.clear()
    
    position, velocity = float(obs[0]), float(obs[1])
    
    # Draw mountain
    xs = np.linspace(-1.2, 0.6, 100)
    ys = np.sin(3 * xs) * 0.45 + 0.55
    ax.fill_between(xs, 0, ys, color='green', alpha=0.3)
    ax.plot(xs, ys, 'g-', linewidth=2)
    
    # Draw goal
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
    ax.plot(0.5, np.sin(3 * 0.5) * 0.45 + 0.55, 'r*', markersize=15)
    
    # Draw car
    car_y = np.sin(3 * position) * 0.45 + 0.55
    ax.plot(position, car_y, 'bo', markersize=12)
    
    ax.set_xlim(-1.3, 0.7)
    ax.set_ylim(0, 1.2)
    ax.set_title(f'MountainCar - Step {step}' if step is not None else 'MountainCar')
    
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    
    return image, fig, ax


def render_acrobot_frame(obs, fig=None, ax=None, step=None):
    """Render a single Acrobot frame from observation."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.clear()
    
    cos1, sin1, cos2, sin2, _, _ = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]
    
    # Link lengths
    l1, l2 = 1.0, 1.0
    
    # Joint positions
    p1 = [float(l1 * sin1), -float(l1 * cos1)]
    p2 = [p1[0] + float(l2 * sin2) * float(cos1) + float(l2 * cos2) * float(sin1),
          p1[1] - float(l2 * sin2) * float(sin1) + float(l2 * cos2) * float(cos1)]
    
    # Simplified joint 2 calculation
    theta1 = np.arctan2(float(sin1), float(cos1))
    theta2 = np.arctan2(float(sin2), float(cos2))
    p2 = [p1[0] + l2 * np.sin(theta1 + theta2),
          p1[1] - l2 * np.cos(theta1 + theta2)]
    
    # Draw links
    ax.plot([0, p1[0]], [0, p1[1]], 'b-', linewidth=4)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=4)
    
    # Draw joints
    ax.plot(0, 0, 'ko', markersize=10)
    ax.plot(p1[0], p1[1], 'ko', markersize=8)
    ax.plot(p2[0], p2[1], 'go', markersize=8)
    
    # Draw goal line
    ax.axhline(y=l1, color='green', linestyle='--', alpha=0.5)
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title(f'Acrobot - Step {step}' if step is not None else 'Acrobot')
    
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    
    return image, fig, ax


# ============================================================================
# Scoring Function
# ============================================================================

def make_scoring_fn(env, env_params, policy, param_template, episode_length, env_name, num_evals=10):
    """Create scoring function that returns fitness and behavioral descriptor.
    
    Evaluates each individual with num_evals trials:
    - Returns fitness from FIRST trial only (for selection)
    - Returns mean fitness across all trials (for logging/tracking)
    """
    
    def evaluate_single(flat_params, eval_key):
        """Evaluate a single individual and return fitness + descriptor."""
        params = unflatten_params(flat_params, param_template)
        obs, state = env.reset(eval_key, env_params)
        
        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
            logits = policy.apply(params, obs)
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
    
    vmapped_eval = jax.vmap(evaluate_single)
    
    @jax.jit
    def scoring_fn(flat_genotypes, key):
        """Returns (fitness_for_selection, descriptors, mean_fitness_for_logging)."""
        pop_size = flat_genotypes.shape[0]
        
        # Always evaluate with num_evals trials per individual
        all_keys = random.split(key, pop_size * num_evals)
        flat_params_repeated = jnp.repeat(flat_genotypes, num_evals, axis=0)
        all_fitnesses, all_descriptors = vmapped_eval(flat_params_repeated, all_keys)
        all_fitnesses = all_fitnesses.reshape(pop_size, num_evals)
        all_descriptors = all_descriptors.reshape(pop_size, num_evals, -1)
        
        # Fitness for selection: use FIRST trial only
        fitnesses = all_fitnesses[:, 0]
        descriptors = all_descriptors[:, 0, :]
        
        # Mean fitness for logging/tracking
        mean_fitnesses = jnp.mean(all_fitnesses, axis=1)
        
        return fitnesses, descriptors, mean_fitnesses
    
    return scoring_fn


def make_eval_best_fn(env, env_params, policy, param_template, episode_length, num_eval_trials=10):
    """Create function to evaluate a single individual with multiple trials for logging."""
    
    def evaluate_single(flat_params, eval_key):
        params = unflatten_params(flat_params, param_template)
        obs, state = env.reset(eval_key, env_params)
        
        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
            logits = policy.apply(params, obs)
            action = jnp.argmax(logits)
            
            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)
            
            total_reward = total_reward + reward * (1.0 - done_flag)
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
            
            return (next_obs, next_state, total_reward, done_flag, key), None
        
        (_, _, total_reward, _, _), _ = jax.lax.scan(
            step_fn, (obs, state, 0.0, 0.0, eval_key), None, length=episode_length
        )
        return total_reward
    
    vmapped_eval = jax.vmap(evaluate_single, in_axes=(None, 0))
    
    @jax.jit
    def eval_best_fn(flat_params, key):
        """Evaluate single individual with multiple trials, return mean fitness."""
        keys = random.split(key, num_eval_trials)
        fitnesses = vmapped_eval(flat_params, keys)
        return jnp.mean(fitnesses)
    
    return eval_best_fn


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
        total_reward += float(reward)
        obs_list.append(np.array(obs))
        
        if bool(done):
            if verbose:
                print(f"  Episode ended at step {step}, reward={total_reward:.2f}")
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
    parser = argparse.ArgumentParser(description='DNS on Gymnax (Non-Continual)')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'])
    parser.add_argument('--num_generations', type=int, default=None)
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--iso_sigma', type=float, default=None)
    parser.add_argument('--line_sigma', type=float, default=None)
    parser.add_argument('--num_evals', type=int, default=None)
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
    batch_size = args.batch_size if args.batch_size is not None else max(1, pop_size // 2)
    batch_size = min(batch_size, pop_size)
    hidden_dims = cfg["hidden_dims"]
    episode_length = cfg["episode_length"]
    num_evals = args.num_evals or cfg["num_evals"]
    k = args.k or cfg["k"]
    iso_sigma = args.iso_sigma or cfg["iso_sigma"]
    line_sigma = args.line_sigma or cfg["line_sigma"]
    
    output_dir = args.output_dir or f"projects/gymnax/dns_{env_name}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, "train.log")
    sys.stdout = Tee(log_file)
    
    print("=" * 60)
    print(f"DNS on {env_name} (Non-Continual)")
    print("=" * 60)
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}, Batch: {batch_size}")
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
        'hidden_dims': hidden_dims,
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"dns_{env_name}_trial{trial}", reinit=True)
    
    # Initialize population
    key, pop_key = jax.random.split(key)
    population = random.normal(pop_key, (pop_size, num_params)) * 0.1
    
    # Initial evaluation
    print(f"\nEvaluating initial population...")
    key, eval_key = jax.random.split(key)
    fitnesses, descriptors, mean_fitnesses = scoring_fn(population, eval_key)
    novelties = _compute_dominated_novelty(fitnesses, descriptors, k)
    
    print(f"  Initial best fitness (selection): {float(jnp.max(fitnesses)):.2f}")
    print(f"  Initial best fitness (mean 10): {float(jnp.max(mean_fitnesses)):.2f}")
    
    # Training loop
    best_overall_fitness = -float('inf')  # Tracks mean-of-10 fitness
    best_params = None
    start_time = time.time()
    
    print(f"\nStarting training...")
    
    for gen in range(num_generations):
        # Generate offspring
        key, var_key = jax.random.split(key)
        offspring = isoline_variation(population, var_key, iso_sigma, line_sigma, batch_size)
        
        # Evaluate offspring (fitness for selection, mean_fitness for tracking)
        key, eval_key = jax.random.split(key)
        offspring_fitnesses, offspring_descriptors, offspring_mean_fitnesses = scoring_fn(offspring, eval_key)
        
        # DNS selection uses single-trial fitness
        population, fitnesses, descriptors, novelties = dns_selection(
            population, fitnesses, descriptors,
            offspring, offspring_fitnesses, offspring_descriptors,
            pop_size, k
        )
        
        # Re-evaluate selected population to get mean fitnesses for tracking
        key, reeval_key = jax.random.split(key)
        _, _, mean_fitnesses = scoring_fn(population, reeval_key)
        
        # Copy to host for numpy operations
        mean_fitness_host = jax.device_get(mean_fitnesses)
        population_host = jax.device_get(population)
        
        # Track using mean-of-10 fitness
        gen_best = float(np.max(mean_fitness_host))
        gen_mean = float(np.mean(mean_fitness_host))
        best_idx = int(np.argmax(mean_fitness_host))
        
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
            "mean_novelty": mean_novelty,
        })
        
        if gen % args.log_interval == 0 or gen == num_generations - 1:
            print(f"Gen {gen:4d} | Best: {gen_best:8.2f} | Mean: {gen_mean:8.2f} | Overall: {best_overall_fitness:8.2f} | Nov: {mean_novelty:.3f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Best overall (training): {best_overall_fitness:.2f}")
    
    # Re-evaluate final population to get accurate mean fitness
    print(f"\nRe-evaluating final population (mean of 10)...")
    key, reeval_key = jax.random.split(key)
    _, _, final_mean_fitnesses = scoring_fn(population, reeval_key)
    final_best_idx = int(jnp.argmax(final_mean_fitnesses))
    population_host = jax.device_get(population)
    best_params = population_host[final_best_idx].copy()
    print(f"  Re-evaluated best (mean of 10): {float(final_mean_fitnesses[final_best_idx]):.2f}")
    
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
    ckpt_path = os.path.join(output_dir, f"dns_{env_name}_best.pkl")
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
        
        if 'CartPole' in env_name:
            render_fn = render_cartpole_frame
        elif 'MountainCar' in env_name:
            render_fn = render_mountaincar_frame
        elif 'Acrobot' in env_name:
            render_fn = render_acrobot_frame
        else:
            render_fn = render_cartpole_frame
        
        for gif_idx in range(10):
            key, gif_key = jax.random.split(key)
            obs_list, total_reward = rollout_for_gif(
                env, env_params, policy, best_params, param_template, episode_length, gif_key
            )
            
            obs_to_render = obs_list[::2]  # Skip every other frame
            
            frames = []
            fig, ax = None, None
            for idx, obs in enumerate(obs_to_render):
                step = idx * 2  # Actual step in episode
                frame, fig, ax = render_fn(obs, fig, ax, step=step)
                frames.append(frame)
            plt.close(fig)
            
            gif_path = os.path.join(gifs_dir, f"eval_{gif_idx:02d}_reward{total_reward:.0f}.gif")
            imageio.mimsave(gif_path, frames, fps=30, loop=0)
        
        print(f"Saved 10 GIFs to: {gifs_dir}")
    except Exception as e:
        print(f"Warning: Failed to save GIF: {e}")
    
    wandb.finish()
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
