"""
Train Dominated Novelty Search (DNS) on MuJoCo Playground with Continual Learning (physics changes).

This script trains a DNS population across multiple tasks where physics (friction) is modified between tasks.
The population is preserved across tasks (no reset), enabling continual learning.

DNS combines novelty search with fitness selection using Pareto dominance.

Usage:
    python train_DNS_cheetah_continual.py --env CheetahRun --num_tasks 30
    python train_DNS_cheetah_continual.py --env WalkerRun --task_mod friction
"""

import argparse
import functools
import os
import sys
from typing import Any, Tuple

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

# Set MuJoCo to use EGL for headless rendering (no display required)
os.environ["MUJOCO_GL"] = "egl"

# Now import JAX and other libraries
import jax
import jax.numpy as jnp
from jax import random, flatten_util
import flax.linen as nn
from mujoco import mjx
from mujoco_playground import registry
import time
import pickle
import wandb
import numpy as np
import imageio


# ============================================================================
# Configuration
# ============================================================================

POLICY_HIDDEN_LAYER_SIZES = (128, 128)


# ============================================================================
# MLP Policy Network
# ============================================================================

class MLPPolicy(nn.Module):
    """Simple MLP policy network."""
    hidden_dims: tuple = (128, 128)
    action_dim: int = 6
    
    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)  # Actions bounded to [-1, 1]
        return x


def get_flat_params(params):
    """Flatten nested parameter dict to a single vector."""
    flat_params, _ = flatten_util.ravel_pytree(params)
    return flat_params


def unflatten_params(flat_params, param_template):
    """Unflatten a vector back to nested parameter dict."""
    _, unravel_fn = flatten_util.ravel_pytree(param_template)
    return unravel_fn(flat_params)


# ============================================================================
# DNS Algorithm Implementation
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
# Environment utilities
# ============================================================================

def create_modified_env(env_name, task_mod='friction', multiplier=1.0):
    """Create environment with modified physics (gravity or friction)."""
    env = registry.load(env_name)
    
    if task_mod == 'gravity':
        gravity_z = -9.81 * multiplier
        env.mj_model.opt.gravity[2] = gravity_z
        print(f"  Gravity set: multiplier={multiplier}, gravity_z={gravity_z:.2f}")
    elif task_mod == 'friction':
        env.mj_model.geom_friction[:] *= multiplier
        print(f"  Friction set: multiplier={multiplier}")
    
    env._mjx_model = mjx.put_model(env.mj_model)
    return env


def sample_task_multipliers(num_tasks, default_mult=1.0, low_mult=0.2, high_mult=5.0):
    """Generate multiplier values for each task, cycling through three values."""
    multiplier_cycle = [default_mult, low_mult, high_mult]
    return [multiplier_cycle[i % 3] for i in range(num_tasks)]


# ============================================================================
# Evaluation function (JIT-compiled)
# ============================================================================

def make_scoring_fn(env, policy, param_template, episode_length, num_evals=1):
    """Create a JIT-compiled scoring function for a specific environment.
    
    This follows the same pattern as the example file to ensure GPU acceleration.
    If num_evals > 1, each individual is evaluated multiple times and fitness is averaged.
    """
    # Get JIT-compiled env functions
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    def evaluate_single(flat_params, eval_key):
        """Evaluate a single individual."""
        params = unflatten_params(flat_params, param_template)
        
        reset_key, step_key = random.split(eval_key)
        state = jit_reset(reset_key)
        
        def step_fn(carry, _):
            state, total_reward = carry
            obs = state.obs
            action = policy.apply(params, obs)
            next_state = jit_step(state, action)
            reward = next_state.reward
            total_reward = total_reward + reward
            return (next_state, total_reward), None
        
        (final_state, total_reward), _ = jax.lax.scan(
            step_fn,
            (state, 0.0),
            None,
            length=episode_length,
        )
        
        # Use final position as descriptor (first 2 dims of obs)
        descriptor = final_state.obs[:2]
        
        return total_reward, descriptor
    
    # Vectorize evaluation across population
    vmapped_eval = jax.vmap(evaluate_single)
    
    @jax.jit
    def scoring_fn(flat_genotypes, key):
        """JIT-compiled scoring function for the population."""
        pop_size = flat_genotypes.shape[0]
        if num_evals == 1:
            keys = random.split(key, pop_size)
            fitnesses, descriptors = vmapped_eval(flat_genotypes, keys)
        else:
            # Multiple evaluations per individual, average the results
            all_keys = random.split(key, pop_size * num_evals)
            flat_params_repeated = jnp.repeat(flat_genotypes, num_evals, axis=0)
            all_fitnesses, all_descriptors = vmapped_eval(flat_params_repeated, all_keys)
            all_fitnesses = all_fitnesses.reshape(pop_size, num_evals)
            all_descriptors = all_descriptors.reshape(pop_size, num_evals, -1)
            fitnesses = jnp.mean(all_fitnesses, axis=1)
            descriptors = jnp.mean(all_descriptors, axis=1)
        return fitnesses, descriptors
    
    return scoring_fn


# ============================================================================
# GIF saving utilities
# ============================================================================

def save_task_gifs(env, policy, best_params, task_idx, task_label, 
                   output_dir, episode_length=1000, num_gifs=3, seed=42):
    """Save GIFs of the best individual at end of task."""
    try:
        task_gifs_dir = os.path.join(output_dir, "gifs", f"task_{task_idx:02d}_{task_label}")
        os.makedirs(task_gifs_dir, exist_ok=True)
        
        # JIT-compiled trajectory rollout using scan (much faster than Python loop)
        @jax.jit
        def rollout_trajectory(key):
            state = env.reset(key)
            def step_fn(carry, _):
                state, total_reward, done_flag = carry
                action = policy.apply(best_params, state.obs)
                next_state = env.step(state, action)
                total_reward = total_reward + next_state.reward * (1.0 - done_flag)
                done_flag = jnp.maximum(done_flag, next_state.done)
                return (next_state, total_reward, done_flag), state
            (final_state, total_reward, _), trajectory_states = jax.lax.scan(
                step_fn, (state, 0.0, 0.0), None, length=episode_length
            )
            return total_reward, trajectory_states
        
        for gif_idx in range(num_gifs):
            key = jax.random.key(seed + task_idx * 1000 + gif_idx * 37)
            total_reward, trajectory_states = rollout_trajectory(key)
            total_reward = float(total_reward)
            
            # Convert stacked states to list for rendering (every 2nd frame)
            trajectory_list = [jax.tree.map(lambda x: x[i], trajectory_states)
                               for i in range(0, episode_length, 2)]
            
            images = env.render(trajectory_list, height=240, width=320, camera="side")
            gif_path = os.path.join(task_gifs_dir, f"trial{gif_idx}_reward{total_reward:.0f}.gif")
            imageio.mimsave(gif_path, images, fps=30, loop=0)
        
        print(f"  Saved {num_gifs} GIFs to: {task_gifs_dir}")
    except Exception as e:
        print(f"  Warning: Failed to save GIFs for task {task_idx}: {e}")


# ============================================================================
# Diversity metrics
# ============================================================================

def compute_fitness_diversity(fitnesses):
    """Compute fitness diversity as standard deviation."""
    return float(jnp.std(fitnesses))


def compute_genomic_diversity(genotypes, sample_size=32):
    """Compute genomic diversity using sampling to avoid OOM.
    
    Full pairwise computation is O(pop_size^2 * num_params) which causes OOM
    with large networks. Instead sample a subset of individuals.
    """
    pop_size = genotypes.shape[0]
    if pop_size < 2:
        return 0.0
    
    # Sample a subset to avoid OOM
    actual_sample = min(sample_size, pop_size)
    indices = np.random.choice(pop_size, size=actual_sample, replace=False)
    sampled = genotypes[indices]
    
    # Normalize on CPU to save GPU memory
    sampled_np = np.array(sampled)
    g_min = np.min(sampled_np, axis=0, keepdims=True)
    g_max = np.max(sampled_np, axis=0, keepdims=True)
    g_range = np.maximum(g_max - g_min, 1e-8)
    norm_genotypes = (sampled_np - g_min) / g_range
    
    # Compute pairwise distances on sampled subset (on CPU)
    diffs = norm_genotypes[:, None, :] - norm_genotypes[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    
    # Average of upper triangle
    mask = np.triu(np.ones((actual_sample, actual_sample)), k=1)
    mean_dist = np.sum(distances * mask) / np.sum(mask)
    
    return float(mean_dist)


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='DNS Continual Learning on MuJoCo Playground')
    parser.add_argument('--env', type=str, default='CheetahRun')
    parser.add_argument('--num_tasks', type=int, default=30)
    parser.add_argument('--episodes_per_task', type=int, default=200)
    parser.add_argument('--pop_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--num_evals', type=int, default=3,
                        help='Number of evaluations per individual to average fitness')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--iso_sigma', type=float, default=0.05)
    parser.add_argument('--line_sigma', type=float, default=0.5)
    parser.add_argument('--task_mod', type=str, default='friction', choices=['gravity', 'friction'])
    parser.add_argument('--gravity_default_mult', type=float, default=1.0)
    parser.add_argument('--gravity_low_mult', type=float, default=0.2)
    parser.add_argument('--gravity_high_mult', type=float, default=5.0)
    parser.add_argument('--friction_default_mult', type=float, default=1.0)
    parser.add_argument('--friction_low_mult', type=float, default=0.2)
    parser.add_argument('--friction_high_mult', type=float, default=5.0)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='continual_neuroevolution_dns')
    parser.add_argument('--log_interval', type=int, default=10)
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    
    env_name = args.env
    pop_size = args.pop_size
    batch_size = args.batch_size if args.batch_size is not None else max(1, pop_size // 2)
    batch_size = min(batch_size, pop_size)
    episodes_per_task = args.episodes_per_task
    num_tasks = args.num_tasks
    episode_length = args.episode_length
    num_evals = args.num_evals
    seed = args.seed
    k = args.k
    iso_sigma = args.iso_sigma
    line_sigma = args.line_sigma
    task_mod = args.task_mod
    log_interval = args.log_interval
    
    # Get multipliers
    if task_mod == 'gravity':
        default_mult = args.gravity_default_mult
        low_mult = args.gravity_low_mult
        high_mult = args.gravity_high_mult
    else:
        default_mult = args.friction_default_mult
        low_mult = args.friction_low_mult
        high_mult = args.friction_high_mult
    
    multiplier_list = sample_task_multipliers(num_tasks, default_mult, low_mult, high_mult)
    num_iterations = num_tasks * episodes_per_task
    
    print("=" * 80)
    print(f"DNS Continual Learning on MuJoCo Playground: {env_name}")
    print("=" * 80)
    print(f"\nContinual Learning Setup:")
    print(f"  Task modification: {task_mod}")
    print(f"  Multipliers: {default_mult}x -> {low_mult}x -> {high_mult}x (cycling)")
    print(f"  Number of tasks: {num_tasks}")
    print(f"  Episodes per task: {episodes_per_task}")
    print(f"  Population size: {pop_size}, Batch size: {batch_size}")
    print(f"  Num evals per individual: {num_evals}")
    print(f"  k (novelty neighbors): {k}")
    
    # Output directory
    mod_suffix = "gravity" if task_mod == "gravity" else "friction"
    output_dir = args.output_dir or f"results_dns_continual_{env_name.lower()}_{mod_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize environment to get dimensions
    base_env = registry.load(env_name)
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    state = base_env.reset(reset_key)
    obs_dim = state.obs.shape[-1]
    action_dim = base_env.action_size
    
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    
    # Initialize wandb
    run_name = args.run_name or f"dns_continual_{env_name}_{task_mod}_seed{seed}"
    wandb.init(
        project=args.wandb_project,
        config={
            "algorithm": "DNS",
            "env_name": env_name,
            "seed": seed,
            "task_mod": task_mod,
            "num_tasks": num_tasks,
            "episodes_per_task": episodes_per_task,
            "episode_length": episode_length,
            "batch_size": batch_size,
            "population_size": pop_size,
            "k": k,
            "iso_sigma": iso_sigma,
            "line_sigma": line_sigma,
        },
        name=run_name,
        reinit=True,
    )
    
    # Initialize policy network
    policy = MLPPolicy(hidden_dims=POLICY_HIDDEN_LAYER_SIZES, action_dim=action_dim)
    key, init_key = jax.random.split(key)
    dummy_obs = jnp.zeros((obs_dim,))
    param_template = policy.init(init_key, dummy_obs)
    
    # Get flattened parameter size
    flat_template = get_flat_params(param_template)
    param_dim = flat_template.shape[0]
    print(f"  Parameter dim: {param_dim}")
    
    # Initialize population
    key, pop_key = jax.random.split(key)
    population = random.normal(pop_key, (pop_size, param_dim)) * 0.1
    
    # Create environments and JIT-compiled scoring functions for each unique multiplier
    unique_multipliers = list(set(multiplier_list))
    envs = {}
    scoring_fns = {}
    for mult in unique_multipliers:
        print(f"  Creating environment for {task_mod}={mult}...")
        envs[mult] = create_modified_env(env_name, task_mod, mult)
        scoring_fns[mult] = make_scoring_fn(envs[mult], policy, param_template, episode_length, num_evals=num_evals)
    
    # Track metrics
    best_fitness_overall = -float('inf')
    best_genotype_overall = None
    best_fitness_per_task = []
    training_metrics_list = []
    
    start_time = time.time()
    iteration = 0
    
    # Initial evaluation
    current_multiplier = multiplier_list[0]
    current_env = envs[current_multiplier]
    current_scoring_fn = scoring_fns[current_multiplier]
    
    print(f"\nEvaluating initial population (JIT compiling, may take a moment)...")
    key, eval_key = jax.random.split(key)
    fitnesses, descriptors = current_scoring_fn(population, eval_key)
    novelties = _compute_dominated_novelty(fitnesses, descriptors, k)
    
    print(f"  Initial best fitness: {float(jnp.max(fitnesses)):.2f}")
    
    # Main training loop
    for task_idx in range(num_tasks):
        current_multiplier = multiplier_list[task_idx]
        current_env = envs[current_multiplier]
        current_scoring_fn = scoring_fns[current_multiplier]
        task_label = f"{task_mod}_{current_multiplier:.2f}".replace(".", "p")
        
        print(f"\n{'='*60}")
        print(f"TASK {task_idx + 1}/{num_tasks} - {task_mod}: {current_multiplier:.2f}")
        print(f"{'='*60}")
        
        # Re-evaluate population on new environment (except first task)
        if task_idx > 0:
            print(f"  Re-evaluating population for {task_mod}={current_multiplier}...")
            key, eval_key = jax.random.split(key)
            fitnesses, descriptors = current_scoring_fn(population, eval_key)
            novelties = _compute_dominated_novelty(fitnesses, descriptors, k)
        
        task_best_fitness = float(jnp.max(fitnesses))
        task_best_idx = int(jnp.argmax(fitnesses))
        task_best_genotype = population[task_best_idx]
        
        # Train for episodes_per_task iterations
        for ep in range(episodes_per_task):
            # Generate offspring
            key, var_key = jax.random.split(key)
            offspring = isoline_variation(population, var_key, iso_sigma, line_sigma, batch_size)
            
            # Evaluate offspring
            key, eval_key = jax.random.split(key)
            offspring_fitnesses, offspring_descriptors = current_scoring_fn(offspring, eval_key)
            
            # DNS selection
            population, fitnesses, descriptors, novelties = dns_selection(
                population, fitnesses, descriptors,
                offspring, offspring_fitnesses, offspring_descriptors,
                pop_size, k
            )
            
            # Update best - always track current generation's best (final gen will be used for eval)
            gen_best_fitness = float(jnp.max(fitnesses))
            task_best_idx = int(jnp.argmax(fitnesses))
            task_best_genotype = population[task_best_idx]
            if gen_best_fitness > task_best_fitness:
                task_best_fitness = gen_best_fitness
            if gen_best_fitness > best_fitness_overall:
                best_fitness_overall = gen_best_fitness
                best_idx = int(jnp.argmax(fitnesses))
                best_genotype_overall = population[best_idx]
            
            # Compute diversity metrics
            fitness_div = compute_fitness_diversity(fitnesses)
            genomic_div = compute_genomic_diversity(population)
            mean_novelty = float(jnp.nanmean(novelties))  # Use nanmean to ignore NaN
            
            # Store metrics
            metrics_entry = {
                'iteration': iteration,
                'task': task_idx,
                'task_iteration': ep,
                'multiplier': current_multiplier,
                'best_fitness': gen_best_fitness,
                'mean_fitness': float(jnp.mean(fitnesses)),
                'fitness_diversity': fitness_div,
                'genomic_diversity': genomic_div,
                'mean_novelty': mean_novelty,
                'task_best_fitness': task_best_fitness,
                'best_overall_fitness': best_fitness_overall,
                'elapsed_time': time.time() - start_time,
            }
            training_metrics_list.append(metrics_entry)
            
            # Log to wandb
            wandb.log({
                "iteration": iteration,
                "task": task_idx,
                "multiplier": current_multiplier,
                "best_fitness": gen_best_fitness,
                "mean_fitness": float(jnp.mean(fitnesses)),
                "fitness_diversity": fitness_div,
                "genomic_diversity": genomic_div,
                "mean_novelty": mean_novelty,
                "task_best_fitness": task_best_fitness,
                "best_overall_fitness": best_fitness_overall,
            })
            
            # Print progress
            if ep % log_interval == 0 or ep == episodes_per_task - 1:
                print(f"Task {task_idx+1} Ep {ep:4d} | "
                      f"Best: {gen_best_fitness:8.2f} | "
                      f"Mean: {float(jnp.mean(fitnesses)):8.2f} | "
                      f"Novelty: {mean_novelty:6.2f} | "
                      f"Task Best: {task_best_fitness:8.2f}")
            
            iteration += 1
        
        # End of task
        print(f"\nTask {task_idx + 1} complete! Best fitness: {task_best_fitness:.2f}")
        best_fitness_per_task.append(task_best_fitness)
        
        # Use the actual task best genotype (saved when task_best_fitness was updated)
        best_genotype = task_best_genotype
        best_fitness_reported = task_best_fitness
        
        # Verify by re-evaluating (catches any alignment issues)
        key, verify_key = jax.random.split(key)
        verify_fitness, _ = current_scoring_fn(best_genotype[None, :], verify_key)
        verify_fitness = float(verify_fitness[0])
        print(f"  Best individual: idx={best_idx}, reported_fitness={best_fitness_reported:.2f}, "
              f"verified_fitness={verify_fitness:.2f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"task_{task_idx:02d}_{task_label}.pkl")
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'population': np.array(population),
                'fitnesses': np.array(fitnesses),
                'descriptors': np.array(descriptors),
                'best_genotype': np.array(best_genotype),
                'best_idx': best_idx,
                'best_fitness_reported': best_fitness_reported,
                'best_fitness_verified': verify_fitness,
                'task_idx': task_idx,
                'task_mod': task_mod,
                'multiplier': current_multiplier,
                'task_best_fitness': task_best_fitness,
                'overall_best_fitness': best_fitness_overall,
                'iteration': iteration,
            }, f)
        print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save GIFs
        best_params = unflatten_params(best_genotype, param_template)
        save_task_gifs(
            current_env, policy, best_params, task_idx, task_label,
            output_dir, episode_length, num_gifs=3, seed=seed
        )
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "=" * 60)
    print("Continual Learning Complete!")
    print("=" * 60)
    print(f"  Environment: {env_name}")
    print(f"  Total tasks: {num_tasks}")
    print(f"  Total iterations: {iteration}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Best fitness achieved: {best_fitness_overall:.2f}")
    
    # Save best policy
    if best_genotype_overall is not None:
        best_params = unflatten_params(best_genotype_overall, param_template)
        save_path = os.path.join(output_dir, f"best_{env_name.lower()}_continual_{mod_suffix}_policy.pkl")
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'params': best_params,
                'flat_params': np.array(best_genotype_overall),
                'fitness': float(best_fitness_overall),
                'config': {
                    'env_name': env_name,
                    'hidden_dims': POLICY_HIDDEN_LAYER_SIZES,
                    'obs_dim': obs_dim,
                    'action_dim': action_dim,
                    'num_tasks': num_tasks,
                    'task_mod': task_mod,
                    'task_multipliers': multiplier_list,
                }
            }, f)
        print(f"  Best policy saved to: {save_path}")
    
    # Save training metrics
    if training_metrics_list:
        import csv
        csv_path = os.path.join(output_dir, "training_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(training_metrics_list[0].keys()))
            writer.writeheader()
            writer.writerows(training_metrics_list)
        print(f"  Training metrics saved to: {csv_path}")
    
    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
