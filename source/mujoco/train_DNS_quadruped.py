"""
Train DNS (Dominated Novelty Search) on Go1 quadruped with a single leg damage (non-continual).

DNS combines novelty search with fitness selection using Pareto dominance.

Usage:
    python train_DNS_quadruped.py --leg FR --gpus 0
    python train_DNS_quadruped.py --leg FL --gpus 0
    python train_DNS_quadruped.py --leg RR --gpus 0
    python train_DNS_quadruped.py --leg RL --gpus 0
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

os.environ["MUJOCO_GL"] = "egl"

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
# Leg damage configuration
# ============================================================================

LEG_ACTION_INDICES = {
    'FR': [0, 1, 2], 'FL': [3, 4, 5], 'RR': [6, 7, 8], 'RL': [9, 10, 11],
}
LEG_QPOS_INDICES = {
    'FR': [7, 8, 9], 'FL': [10, 11, 12], 'RR': [13, 14, 15], 'RL': [16, 17, 18],
}
LEG_QVEL_INDICES = {
    'FR': [6, 7, 8], 'FL': [9, 10, 11], 'RR': [12, 13, 14], 'RL': [15, 16, 17],
}
LOCKED_JOINT_POSITIONS = jnp.array([0.0, 1.2, -2.4])


class LegDamageWrapper:
    """Wrapper that locks a damaged leg in a fixed bent position."""
    
    def __init__(self, env, damaged_leg):
        self._env = env
        self._damaged_leg = damaged_leg
        
        self._action_mask = jnp.ones(env.action_size)
        if damaged_leg is not None:
            action_indices = jnp.array(LEG_ACTION_INDICES[damaged_leg])
            self._action_mask = self._action_mask.at[action_indices].set(0.0)
            self._qpos_indices = jnp.array(LEG_QPOS_INDICES[damaged_leg])
            self._qvel_indices = jnp.array(LEG_QVEL_INDICES[damaged_leg])
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


# ============================================================================
# MLP Policy Network
# ============================================================================

class MLPPolicy(nn.Module):
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
    flat_params, _ = flatten_util.ravel_pytree(params)
    return flat_params


def unflatten_params(flat_params, param_template):
    _, unravel_fn = flatten_util.ravel_pytree(param_template)
    return unravel_fn(flat_params)


def create_policy_network(key, obs_dim, action_dim, hidden_dims=(512, 256, 128)):
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
# Scoring function
# ============================================================================

def make_scoring_fn(env, policy, param_template, episode_length, obs_key='state', num_evals=1):
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    def evaluate_single(flat_params, eval_key):
        params = unflatten_params(flat_params, param_template)
        reset_key, _ = random.split(eval_key)
        state = jit_reset(reset_key)
        
        def step_fn(carry, _):
            state, total_reward, done_flag = carry
            obs = state.obs[obs_key]
            action = policy.apply(params, obs)
            next_state = jit_step(state, action)
            reward = next_state.reward
            total_reward = total_reward + reward * (1.0 - done_flag)
            done_flag = jnp.maximum(done_flag, next_state.done)
            return (next_state, total_reward, done_flag), None
        
        (final_state, total_reward, _), _ = jax.lax.scan(
            step_fn, (state, 0.0, 0.0), None, length=episode_length
        )
        # Use final position as descriptor (first 2 dims of state obs)
        descriptor = final_state.obs[obs_key][:2]
        return total_reward, descriptor
    
    vmapped_eval = jax.vmap(evaluate_single)
    
    # JIT-compiled trajectory collection for fast GIF generation
    @jax.jit
    def rollout_with_trajectory(flat_params, eval_key):
        """Rollout episode and return trajectory states for rendering."""
        params = unflatten_params(flat_params, param_template)
        state = jit_reset(eval_key)
        
        def step_fn(carry, _):
            state, total_reward, done_flag = carry
            obs = state.obs[obs_key]
            action = policy.apply(params, obs)
            next_state = jit_step(state, action)
            reward = next_state.reward
            total_reward = total_reward + reward * (1.0 - done_flag)
            done_flag = jnp.maximum(done_flag, next_state.done)
            return (next_state, total_reward, done_flag), state
        
        (final_state, total_reward, _), trajectory_states = jax.lax.scan(
            step_fn, (state, 0.0, 0.0), None, length=episode_length
        )
        return total_reward, trajectory_states
    
    @jax.jit
    def scoring_fn(flat_genotypes, key):
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
    
    return scoring_fn, rollout_with_trajectory


# ============================================================================
# Diversity metrics
# ============================================================================

def compute_fitness_diversity(fitnesses):
    return float(jnp.std(fitnesses))


def compute_genomic_diversity(genotypes, sample_size=32):
    """Compute genomic diversity using sampling to avoid OOM.
    
    Full pairwise computation is O(pop_size^2 * num_params) which causes OOM
    with large networks. Instead sample a subset of individuals.
    """
    pop_size = genotypes.shape[0]
    if pop_size < 2:
        return 0.0
    
    # Sample a subset to avoid OOM (full pairwise is too memory intensive)
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
    
    mask = np.triu(np.ones((actual_sample, actual_sample)), k=1)
    mean_dist = np.sum(distances * mask) / np.sum(mask)
    
    return float(mean_dist)


# ============================================================================
# Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='DNS on Go1 quadruped (Non-Continual)')
    parser.add_argument('--env', type=str, default='Go1JoystickFlatTerrain')
    parser.add_argument('--leg', type=str, default='FR', choices=['FR', 'FL', 'RR', 'RL', 'NONE'],
                        help='Which leg to damage (NONE for healthy robot)')
    parser.add_argument('--num_generations', type=int, default=500)
    parser.add_argument('--pop_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--k', type=int, default=3, help='Number of neighbors for novelty')
    parser.add_argument('--iso_sigma', type=float, default=0.05)
    parser.add_argument('--line_sigma', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--num_evals', type=int, default=3,
                        help='Number of evaluation episodes per individual')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='continual_neuroevolution_dns')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--hidden_dims', type=str, default='512,256,128')
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    
    env_name = args.env
    damaged_leg = args.leg if args.leg != 'NONE' else None
    leg_label = args.leg  # Keep original for naming
    pop_size = args.pop_size
    batch_size = args.batch_size
    num_generations = args.num_generations
    episode_length = args.episode_length
    seed = args.seed
    trial = args.trial
    k = args.k
    iso_sigma = args.iso_sigma
    line_sigma = args.line_sigma
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(','))
    
    output_dir = args.output_dir or f"projects/mujoco/dns_{env_name}_leg{leg_label}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"DNS on {env_name} (Non-Continual)")
    print("=" * 60)
    print(f"  Damaged leg: {leg_label}")
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}, Batch size: {batch_size}")
    print(f"  k (novelty neighbors): {k}")
    print(f"  Output: {output_dir}")
    
    key = jax.random.key(seed)
    
    # Create environment with leg damage
    base_env = registry.load(env_name)
    env = LegDamageWrapper(base_env, damaged_leg)
    
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)
    obs_dim = state.obs['state'].shape[-1]
    action_dim = env.action_size
    
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create policy
    key, init_key = jax.random.split(key)
    policy, param_template = create_policy_network(init_key, obs_dim, action_dim, hidden_dims)
    flat_params = get_flat_params(param_template)
    num_params = flat_params.shape[0]
    print(f"  Network: {hidden_dims}, {num_params} params")
    
    # Create scoring function and trajectory rollout
    scoring_fn, rollout_with_trajectory = make_scoring_fn(env, policy, param_template, episode_length, num_evals=args.num_evals)
    
    # Initialize wandb
    config = {
        'env': env_name, 'damaged_leg': damaged_leg, 'num_generations': num_generations,
        'pop_size': pop_size, 'batch_size': batch_size, 'k': k,
        'iso_sigma': iso_sigma, 'line_sigma': line_sigma,
        'seed': seed, 'trial': trial, 'num_evals': args.num_evals,
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"dns_{env_name}_leg{leg_label}_trial{trial}", reinit=True)
    
    # Initialize population
    key, pop_key = jax.random.split(key)
    population = random.normal(pop_key, (pop_size, num_params)) * 0.1
    
    # Initial evaluation
    print(f"\nEvaluating initial population...")
    key, eval_key = jax.random.split(key)
    fitnesses, descriptors = scoring_fn(population, eval_key)
    novelties = compute_novelty(descriptors, k)
    
    print(f"  Initial best fitness: {float(jnp.max(fitnesses)):.2f}")
    
    # Training loop
    best_overall_fitness = -float('inf')
    start_time = time.time()
    
    print(f"\nStarting training...")
    
    for gen in range(num_generations):
        # Generate offspring
        key, var_key = jax.random.split(key)
        offspring = isoline_variation(population, var_key, iso_sigma, line_sigma, batch_size)
        
        # Evaluate offspring
        key, eval_key = jax.random.split(key)
        offspring_fitnesses, offspring_descriptors = scoring_fn(offspring, eval_key)
        
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
        
        # Track best from current generation (will use final generation for eval)
        final_best_fitness = gen_best
        final_best_params = population_host[best_idx].copy()
        
        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best
        
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
    print(f"  Cached best overall: {best_overall_fitness:.2f}")
    
    # Re-evaluate final population to get accurate fitness (not stale cached values)
    print(f"\nRe-evaluating final population for accurate fitness...")
    key, reeval_key = jax.random.split(key)
    final_fitnesses, _ = scoring_fn(population, reeval_key)
    final_best_idx = int(jnp.argmax(final_fitnesses))
    population_host = jax.device_get(population)
    best_params = population_host[final_best_idx].copy()
    best_fitness = float(final_fitnesses[final_best_idx])
    print(f"  Re-evaluated best fitness: {best_fitness:.2f}")
    
    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"dns_{env_name}_leg{leg_label}_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': np.array(best_params),
            'param_template': param_template,
            'best_fitness': best_fitness,
            'damaged_leg': damaged_leg,
            'config': config,
        }, f)
    print(f"Saved: {ckpt_path}")
    
    # Verify best params fitness
    print(f"\nVerifying final generation's best params...")
    key, verify_key = jax.random.split(key)
    verify_fitness, _ = scoring_fn(best_params[None, :], verify_key)
    print(f"  Final gen reported best: {best_fitness:.2f}")
    print(f"  Verification fitness:    {float(verify_fitness[0]):.2f}")
    
    # Save GIFs using JIT-compiled rollout (much faster than Python loop)
    try:
        gifs_dir = os.path.join(output_dir, "gifs")
        os.makedirs(gifs_dir, exist_ok=True)
        
        print(f"\nSaving {3} evaluation GIFs...")
        for gif_idx in range(3):
            key, gif_key = jax.random.split(key)
            # Use JIT-compiled rollout for fast trajectory collection
            total_reward, trajectory_states = rollout_with_trajectory(best_params, gif_key)
            total_reward = float(total_reward)
            
            # Convert stacked states to list for rendering (use every 4th frame for speed)
            trajectory_list = [jax.tree.map(lambda x: x[i], trajectory_states) 
                               for i in range(0, episode_length, 4)]
            
            images = env.render(trajectory_list, height=240, width=320, camera="track")
            gif_path = os.path.join(gifs_dir, f"trial{gif_idx}_reward{total_reward:.0f}.gif")
            imageio.mimsave(gif_path, images, fps=30, loop=0)
            print(f"  GIF {gif_idx}: reward={total_reward:.2f}")
        
        print(f"Saved GIFs to: {gifs_dir}")
    except Exception as e:
        print(f"Warning: Failed to save GIFs: {e}")
        import traceback
        traceback.print_exc()
    
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
