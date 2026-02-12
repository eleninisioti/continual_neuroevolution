"""
Train Dominated Novelty Search (DNS) on Go1 with Continual Learning (leg damage).

This script trains a DNS population across multiple tasks where different legs are damaged.
The population is preserved across tasks (no reset), enabling continual learning.

DNS combines novelty search with fitness selection using Pareto dominance.

Go1 action structure (12 actions total):
  FR (Front Right): indices 0-2 (hip, thigh, calf)
  FL (Front Left): indices 3-5 (hip, thigh, calf)
  RR (Rear Right): indices 6-8 (hip, thigh, calf)
  RL (Rear Left): indices 9-11 (hip, thigh, calf)

Usage:
    python train_DNS_quadruped_continual.py --gpus 4
    python train_DNS_quadruped_continual.py --num_tasks 20 --episodes_per_task 50
"""

import argparse
import functools
import os
import sys
import json
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
from mujoco_playground import registry
import time
import pickle
import wandb
import numpy as np
import imageio


# ============================================================================
# Configuration
# ============================================================================

POLICY_HIDDEN_LAYER_SIZES = (512, 256, 128)

# Leg action indices
LEG_ACTION_INDICES = {
    None: [],        # Healthy (no damage)
    0: [0, 1, 2],    # Front Right (FR)
    1: [3, 4, 5],    # Front Left (FL)
    2: [6, 7, 8],    # Rear Right (RR)
    3: [9, 10, 11],  # Rear Left (RL)
}

# Leg qpos indices (joint positions in state.data.qpos)
LEG_QPOS_INDICES = {
    None: [],        # Healthy (no damage)
    0: [7, 8, 9],    # Front Right (FR)
    1: [10, 11, 12], # Front Left (FL)
    2: [13, 14, 15], # Rear Right (RR)
    3: [16, 17, 18], # Rear Left (RL)
}

# Leg qvel indices (joint velocities in state.data.qvel)
LEG_QVEL_INDICES = {
    None: [],        # Healthy (no damage)
    0: [6, 7, 8],    # Front Right (FR)
    1: [9, 10, 11],  # Front Left (FL)
    2: [12, 13, 14], # Rear Right (RR)
    3: [15, 16, 17], # Rear Left (RL)
}

# Locked joint positions (bent/tucked position)
LOCKED_JOINT_POSITIONS = jnp.array([0.0, 1.2, -2.4])

LEG_NAMES = {None: 'HEALTHY', 0: 'FR', 1: 'FL', 2: 'RR', 3: 'RL'}


# ============================================================================
# Leg Damage Wrapper
# ============================================================================

class LegDamageWrapper:
    """Wrapper that locks a damaged leg in a fixed bent position."""
    
    def __init__(self, env, damaged_leg_idx):
        self._env = env
        self._damaged_leg = damaged_leg_idx
        
        self._action_mask = jnp.ones(env.action_size)
        if damaged_leg_idx is not None:
            action_indices = jnp.array(LEG_ACTION_INDICES[damaged_leg_idx])
            self._action_mask = self._action_mask.at[action_indices].set(0.0)
            self._qpos_indices = jnp.array(LEG_QPOS_INDICES[damaged_leg_idx])
            self._qvel_indices = jnp.array(LEG_QVEL_INDICES[damaged_leg_idx])
        else:
            self._qpos_indices = None
            self._qvel_indices = None
    
    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def _lock_leg_joints(self, state):
        if self._damaged_leg is None:
            return state
        new_qpos = state.data.qpos.at[self._qpos_indices].set(LOCKED_JOINT_POSITIONS)
        new_qvel = state.data.qvel.at[self._qvel_indices].set(0.0)
        new_data = state.data.replace(qpos=new_qpos, qvel=new_qvel)
        return state.replace(data=new_data)
    
    def step(self, state, action):
        masked_action = action * self._action_mask
        next_state = self._env.step(state, masked_action)
        return self._lock_leg_joints(next_state)
    
    def reset(self, rng):
        state = self._env.reset(rng)
        return self._lock_leg_joints(state)


def generate_random_leg_sequence(rng_key, num_tasks: int, avoid_consecutive: bool = True):
    """Generate a random sequence of legs to damage."""
    sequence = []
    
    for i in range(num_tasks):
        rng_key, subkey = random.split(rng_key)
        
        if avoid_consecutive and len(sequence) > 0:
            last_leg = sequence[-1]
            available_legs = [leg for leg in range(4) if leg != last_leg]
            idx = int(random.randint(subkey, (), 0, len(available_legs)))
            leg = available_legs[idx]
        else:
            leg = int(random.randint(subkey, (), 0, 4))
        
        sequence.append(leg)
    
    return sequence


# ============================================================================
# MLP Policy Network
# ============================================================================

class MLPPolicy(nn.Module):
    """MLP policy network for locomotion environments."""
    hidden_dims: tuple = (512, 256, 128)
    action_dim: int = 12
    
    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.silu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
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

def compute_novelty(descriptors: jnp.ndarray, k: int = 3) -> jnp.ndarray:
    """Compute novelty scores as average distance to k nearest neighbors."""
    # Handle NaN descriptors by replacing with zeros (will get low novelty)
    descriptors = jnp.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute pairwise distances
    diffs = descriptors[:, None, :] - descriptors[None, :, :]
    distances = jnp.linalg.norm(diffs, axis=-1)
    
    # Set self-distances to inf so they're not selected as neighbors
    distances = distances + jnp.eye(distances.shape[0]) * 1e10
    
    # Get k smallest distances for each individual
    k_nearest_distances = jnp.sort(distances, axis=1)[:, :k]
    
    # Novelty is average of k nearest neighbor distances
    novelty = jnp.mean(k_nearest_distances, axis=1)
    
    # Replace any remaining NaN with 0 (low novelty score)
    novelty = jnp.nan_to_num(novelty, nan=0.0, posinf=0.0, neginf=0.0)
    
    return novelty


def dns_selection(
    genotypes: jnp.ndarray,
    fitnesses: jnp.ndarray,
    descriptors: jnp.ndarray,
    new_genotypes: jnp.ndarray,
    new_fitnesses: jnp.ndarray,
    new_descriptors: jnp.ndarray,
    population_size: int,
    k: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """DNS selection: combine parent and offspring, select based on combined fitness+novelty score."""
    
    # Combine parents and offspring
    combined_genotypes = jnp.concatenate([genotypes, new_genotypes], axis=0)
    combined_fitnesses = jnp.concatenate([fitnesses, new_fitnesses], axis=0)
    combined_descriptors = jnp.concatenate([descriptors, new_descriptors], axis=0)
    
    # Compute novelty for combined population
    combined_novelties = compute_novelty(combined_descriptors, k)
    
    # Handle NaN fitnesses (assign very low fitness)
    combined_fitnesses = jnp.nan_to_num(combined_fitnesses, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Normalize fitness to [0, 1]
    f_min, f_max = jnp.min(combined_fitnesses), jnp.max(combined_fitnesses)
    f_range = jnp.maximum(f_max - f_min, 1e-8)
    norm_fitness = (combined_fitnesses - f_min) / f_range
    
    # Normalize novelty to [0, 1]
    n_min, n_max = jnp.min(combined_novelties), jnp.max(combined_novelties)
    n_range = jnp.maximum(n_max - n_min, 1e-8)
    norm_novelty = (combined_novelties - n_min) / n_range
    
    # Combined score for selection (equal weight to fitness and novelty)
    # Handle any remaining NaN in scores
    combined_scores = jnp.nan_to_num(norm_fitness + norm_novelty, nan=-1e6)
    
    # Select top population_size individuals by combined score
    selected_indices = jnp.argsort(combined_scores)[-population_size:]
    
    # Extract selected individuals
    selected_genotypes = combined_genotypes[selected_indices]
    selected_fitnesses = combined_fitnesses[selected_indices]
    selected_descriptors = combined_descriptors[selected_indices]
    selected_novelties = combined_novelties[selected_indices]
    
    return selected_genotypes, selected_fitnesses, selected_descriptors, selected_novelties


def isoline_variation(
    genotypes: jnp.ndarray,
    key: jnp.ndarray,
    iso_sigma: float = 0.05,
    line_sigma: float = 0.5,
    batch_size: int = 256,
) -> jnp.ndarray:
    """Isoline variation operator for generating offspring."""
    pop_size = genotypes.shape[0]
    param_dim = genotypes.shape[1]
    
    # Sample parent indices
    key, k1, k2, k3, k4 = random.split(key, 5)
    parent1_indices = random.randint(k1, (batch_size,), 0, pop_size)
    parent2_indices = random.randint(k2, (batch_size,), 0, pop_size)
    
    # Get parent genotypes
    parent1 = genotypes[parent1_indices]
    parent2 = genotypes[parent2_indices]
    
    # Isoline crossover
    alpha = random.uniform(k3, (batch_size, param_dim), minval=-line_sigma, maxval=1.0 + line_sigma)
    offspring = parent1 + alpha * (parent2 - parent1)
    
    # Gaussian mutation
    noise = random.normal(k4, (batch_size, param_dim)) * iso_sigma
    offspring = offspring + noise
    
    return offspring


# ============================================================================
# Evaluation function (JIT-compiled)
# ============================================================================

def get_obs(state):
    """Extract observation from state - Go1 returns dict with 'state' key."""
    return state.obs['state']


def make_scoring_fn(env, policy, param_template, episode_length, num_evals=1):
    """Create a JIT-compiled scoring function for a specific environment.
    
    If num_evals > 1, each individual is evaluated multiple times and fitness is averaged.
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    obs_key = 'state'  # Go1 uses dict observations
    
    def evaluate_single(flat_params, eval_key):
        """Evaluate a single individual."""
        params = unflatten_params(flat_params, param_template)
        
        reset_key, step_key = random.split(eval_key)
        state = jit_reset(reset_key)
        
        def step_fn(carry, _):
            state, total_reward, done_flag = carry
            # Go1 returns dict observations with 'state' key
            obs = state.obs[obs_key]
            action = policy.apply(params, obs)
            next_state = jit_step(state, action)
            reward = next_state.reward
            # Don't accumulate reward after done
            total_reward = total_reward + reward * (1.0 - done_flag)
            done_flag = jnp.maximum(done_flag, next_state.done)
            return (next_state, total_reward, done_flag), None
        
        (final_state, total_reward, _), _ = jax.lax.scan(
            step_fn,
            (state, 0.0, 0.0),
            None,
            length=episode_length,
        )
        
        # Use final position as descriptor (first 2 dims of obs)
        # Go1 returns dict observations with 'state' key
        final_obs = final_state.obs[obs_key]
        descriptor = final_obs[:2]
        
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
        
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)
        jit_policy = jax.jit(policy.apply)
        
        for gif_idx in range(num_gifs):
            key = jax.random.key(seed + task_idx * 1000 + gif_idx * 37)
            state = jit_reset(key)
            trajectory = [state]
            total_reward = 0.0
            
            for _ in range(episode_length):
                obs = get_obs(state)
                action = jit_policy(best_params, obs)
                state = jit_step(state, action)
                trajectory.append(state)
                total_reward += float(state.reward)
                if state.done:
                    break
            
            # Render every 2nd frame - use "track" camera for Go1
            images = env.render(trajectory[::2], height=240, width=320, camera="track")
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


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='DNS Continual Learning on Go1 with Leg Damage')
    parser.add_argument('--env', type=str, default='Go1JoystickFlatTerrain')
    parser.add_argument('--num_tasks', type=int, default=20)
    parser.add_argument('--episodes_per_task', type=int, default=50)
    parser.add_argument('--pop_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--num_evals', type=int, default=3,
                        help='Number of evaluations per individual to average fitness')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--iso_sigma', type=float, default=0.05)
    parser.add_argument('--line_sigma', type=float, default=0.5)
    parser.add_argument('--avoid_consecutive', action='store_true', default=True)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, default=None)
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
    batch_size = args.batch_size
    episodes_per_task = args.episodes_per_task
    num_tasks = args.num_tasks
    episode_length = args.episode_length
    num_evals = args.num_evals
    seed = args.seed
    trial = args.trial
    k = args.k
    iso_sigma = args.iso_sigma
    line_sigma = args.line_sigma
    log_interval = args.log_interval
    
    num_iterations = num_tasks * episodes_per_task
    
    # Generate leg damage sequence
    key = jax.random.key(seed)
    key, seq_key = jax.random.split(key)
    leg_sequence = generate_random_leg_sequence(seq_key, num_tasks, args.avoid_consecutive)
    
    print("=" * 80)
    print(f"DNS Continual Learning on Go1 with Leg Damage")
    print("=" * 80)
    print(f"\nContinual Learning Setup:")
    print(f"  Number of tasks: {num_tasks}")
    print(f"  Episodes per task: {episodes_per_task}")
    print(f"  Population size: {pop_size}, Batch size: {batch_size}")
    print(f"  k (novelty neighbors): {k}")
    print(f"\nLeg damage sequence:")
    for i, leg_idx in enumerate(leg_sequence):
        print(f"  Task {i+1}: Damage {LEG_NAMES[leg_idx]} leg")
    
    # Output directory
    output_dir = args.output_dir or f"results_dns_continual_{env_name.lower()}_legdamage"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize environment to get dimensions
    base_env = registry.load(env_name)
    key, reset_key = jax.random.split(key)
    state = base_env.reset(reset_key)
    obs_dim = get_obs(state).shape[-1]
    action_dim = base_env.action_size
    
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    
    # Save config
    config = {
        'env': env_name,
        'num_tasks': num_tasks,
        'episodes_per_task': episodes_per_task,
        'seed': seed,
        'trial': trial,
        'leg_sequence': [LEG_NAMES[l] for l in leg_sequence],
        'leg_action_indices': {LEG_NAMES[k]: v for k, v in LEG_ACTION_INDICES.items()},
        'avoid_consecutive': args.avoid_consecutive,
        'pop_size': pop_size,
        'batch_size': batch_size,
        'k': k,
        'iso_sigma': iso_sigma,
        'line_sigma': line_sigma,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize wandb
    run_name = args.experiment_name or f"dns_go1_legdamage_trial{trial}"
    wandb.init(
        project=args.wandb_project,
        config=config,
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
    
    # Create wrapped environments and scoring functions for each leg
    print(f"\nCreating environments for each leg damage condition...")
    envs = {}
    scoring_fns = {}
    for leg_idx in range(4):  # 4 legs
        print(f"  Creating environment for {LEG_NAMES[leg_idx]} damage...")
        wrapped_env = LegDamageWrapper(registry.load(env_name), leg_idx)
        envs[leg_idx] = wrapped_env
        scoring_fns[leg_idx] = make_scoring_fn(wrapped_env, policy, param_template, episode_length, num_evals=num_evals)
    
    # Track metrics
    best_fitness_overall = -float('inf')
    best_genotype_overall = None
    best_fitness_per_task = []
    training_metrics_list = []
    
    start_time = time.time()
    iteration = 0
    
    # Initial evaluation with first task's leg damage
    current_leg = leg_sequence[0]
    current_env = envs[current_leg]
    current_scoring_fn = scoring_fns[current_leg]
    
    print(f"\nEvaluating initial population (JIT compiling, may take a moment)...")
    key, eval_key = jax.random.split(key)
    fitnesses, descriptors = current_scoring_fn(population, eval_key)
    novelties = compute_novelty(descriptors, k)
    
    print(f"  Initial best fitness: {float(jnp.max(fitnesses)):.2f}")
    
    # Main training loop
    for task_idx in range(num_tasks):
        current_leg = leg_sequence[task_idx]
        current_env = envs[current_leg]
        current_scoring_fn = scoring_fns[current_leg]
        task_label = LEG_NAMES[current_leg]
        
        print(f"\n{'='*60}")
        print(f"TASK {task_idx + 1}/{num_tasks} - Damage: {task_label} leg")
        print(f"{'='*60}")
        
        # Re-evaluate population on new environment (except first task)
        if task_idx > 0:
            print(f"  Re-evaluating population for {task_label} damage...")
            key, eval_key = jax.random.split(key)
            fitnesses, descriptors = current_scoring_fn(population, eval_key)
            novelties = compute_novelty(descriptors, k)
        
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
            mean_novelty = float(jnp.nanmean(novelties))
            
            # Store metrics
            metrics_entry = {
                'iteration': iteration,
                'task': task_idx,
                'task_iteration': ep,
                'damaged_leg': task_label,
                'best_fitness': gen_best_fitness,
                'mean_fitness': float(jnp.mean(fitnesses)),
                'fitness_diversity': fitness_div,
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
                "damaged_leg": task_label,
                "best_fitness": gen_best_fitness,
                "mean_fitness": float(jnp.mean(fitnesses)),
                "fitness_diversity": fitness_div,
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
        
        # Verify by re-evaluating
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
                'damaged_leg': task_label,
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
        save_path = os.path.join(output_dir, f"best_{env_name.lower()}_dns_legdamage_policy.pkl")
        
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
                    'leg_sequence': [LEG_NAMES[l] for l in leg_sequence],
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
