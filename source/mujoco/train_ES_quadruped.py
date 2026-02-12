"""
Train OpenES on Go1 quadruped with a single leg damage (non-continual).

Usage:
    python train_ES_quadruped.py --leg FR --gpus 0
    python train_ES_quadruped.py --leg FL --gpus 0
    python train_ES_quadruped.py --leg RR --gpus 0
    python train_ES_quadruped.py --leg RL --gpus 0
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
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import flax.linen as nn
import optax
from evosax.algorithms import Open_ES
from mujoco_playground import registry
import time
import pickle
import wandb
import json
import numpy as np
import imageio


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
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    return flat_params


def unflatten_params(flat_params, param_template):
    _, unravel_fn = jax.flatten_util.ravel_pytree(param_template)
    return unravel_fn(flat_params)


def create_policy_network(key, obs_dim, action_dim, hidden_dims=(512, 256, 128)):
    policy = MLPPolicy(hidden_dims=hidden_dims, action_dim=action_dim)
    dummy_obs = jnp.zeros((obs_dim,))
    params = policy.init(key, dummy_obs)
    return policy, params


def make_scoring_fn(env, policy, param_template, episode_length, num_evals=3):
    """Create JIT-compiled scoring function."""
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    def evaluate_single(flat_params, eval_key):
        params = unflatten_params(flat_params, param_template)
        reset_key, _ = jax.random.split(eval_key)
        state = jit_reset(reset_key)
        
        def step_fn(carry, _):
            state, total_reward, done_flag = carry
            # Go1 returns dict observations with 'state' key
            obs = state.obs['state']
            action = policy.apply(params, obs)
            next_state = jit_step(state, action)
            reward = next_state.reward
            total_reward = total_reward + reward * (1.0 - done_flag)
            done_flag = jnp.maximum(done_flag, next_state.done)
            return (next_state, total_reward, done_flag), None
        
        (_, total_reward, _), _ = jax.lax.scan(
            step_fn, (state, 0.0, 0.0), None, length=episode_length
        )
        return total_reward
    
    vmapped_eval = jax.vmap(evaluate_single)
    
    def evaluate_with_multi_evals(flat_params, eval_key):
        """Evaluate single individual with multiple rollouts and average."""
        keys = jax.random.split(eval_key, num_evals)
        rewards = jax.vmap(lambda k: evaluate_single(flat_params, k))(keys)
        return jnp.mean(rewards)
    
    # JIT-compiled trajectory collection for fast GIF generation
    @jax.jit
    def rollout_with_trajectory(flat_params, eval_key):
        """Rollout episode and return trajectory states for rendering."""
        params = unflatten_params(flat_params, param_template)
        state = jit_reset(eval_key)
        
        def step_fn(carry, _):
            state, total_reward, done_flag = carry
            obs = state.obs['state']
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
        keys = jax.random.split(key, pop_size)
        if num_evals == 1:
            fitnesses = vmapped_eval(flat_genotypes, keys)
        else:
            # Multi-eval: average over num_evals rollouts per individual
            fitnesses = jax.vmap(evaluate_with_multi_evals)(flat_genotypes, keys)
        return fitnesses
    
    return scoring_fn, rollout_with_trajectory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Go1JoystickFlatTerrain')
    parser.add_argument('--leg', type=str, default='FR', choices=['FR', 'FL', 'RR', 'RL', 'NONE'],
                        help='Which leg to damage (NONE for healthy robot)')
    parser.add_argument('--num_generations', type=int, default=500)
    parser.add_argument('--pop_size', type=int, default=512)
    parser.add_argument('--sigma', type=float, default=0.04)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_evals', type=int, default=3,
                        help='Number of evaluation rollouts to average per individual')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='continual_neuroevolution_es')
    parser.add_argument('--log_interval', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    
    env_name = args.env
    damaged_leg = args.leg if args.leg != 'NONE' else None
    leg_label = args.leg  # Keep original for naming
    pop_size = args.pop_size
    num_generations = args.num_generations
    episode_length = args.episode_length
    seed = args.seed
    trial = args.trial
    hidden_dims = (512, 256, 128)
    
    output_dir = args.output_dir or f"projects/mujoco/es_{env_name}_leg{leg_label}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"OpenES on {env_name} (Non-Continual)")
    print("=" * 60)
    print(f"  Damaged leg: {leg_label}")
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}")
    
    key = jax.random.key(seed)
    
    # Create environment with leg damage
    base_env = registry.load(env_name)
    env = LegDamageWrapper(base_env, damaged_leg)
    
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)
    # Handle dict observations (Go1 returns dict with 'state' key)
    if isinstance(state.obs, dict):
        obs_dim = state.obs['state'].shape[-1]
    else:
        obs_dim = state.obs.shape[-1]
    action_dim = env.action_size
    
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create policy
    key, init_key = jax.random.split(key)
    policy, param_template = create_policy_network(init_key, obs_dim, action_dim, hidden_dims)
    flat_params = get_flat_params(param_template)
    num_params = flat_params.shape[0]
    print(f"  Network: {hidden_dims}, {num_params} params")
    
    # Create scoring function and trajectory rollout
    scoring_fn, rollout_with_trajectory = make_scoring_fn(env, policy, param_template, episode_length, args.num_evals)
    
    # Initialize wandb
    config = {
        'env': env_name, 'damaged_leg': damaged_leg, 'num_generations': num_generations,
        'pop_size': pop_size, 'seed': seed, 'trial': trial,
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"es_{env_name}_leg{leg_label}_trial{trial}", reinit=True)
    
    # Initialize ES with multi-GPU sharding
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(np.array(devices), axis_names=('p',))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())
    parallel_sharding = NamedSharding(mesh, PartitionSpec('p'))
    
    # Ensure pop_size is divisible by num_devices
    if pop_size % num_devices != 0:
        old_pop_size = pop_size
        pop_size = (pop_size // num_devices + 1) * num_devices
        print(f"  Warning: Adjusted pop_size from {old_pop_size} to {pop_size}")
    
    # Custom std schedule (constant at sigma)
    std_schedule = lambda gen: args.sigma
    
    # OpenES uses Adam optimizer
    optimizer = optax.adam(learning_rate=args.learning_rate)
    
    es = Open_ES(
        population_size=pop_size,
        solution=jnp.zeros(num_params),
        std_schedule=std_schedule,
        optimizer=optimizer,
        use_antithetic_sampling=True,
    )
    es_params = jax.device_put(es.default_params, replicate_sharding)
    
    key, init_key = jax.random.split(key)
    es_state = es.init(init_key, jnp.zeros(num_params), es_params)
    es_state = jax.device_put(es_state, replicate_sharding)
    
    # JIT compile ask/tell
    @jax.jit
    def jit_ask(key, state, params):
        population, new_state = es.ask(key, state, params)
        population = jax.device_put(population, parallel_sharding)
        return population, new_state
    
    @jax.jit
    def jit_tell(key, population, fitness, state, params):
        return es.tell(key, population, fitness, state, params)
    
    # Training loop
    best_overall_fitness = -float('inf')
    start_time = time.time()
    
    print(f"\nStarting training...")
    
    for gen in range(num_generations):
        key, ask_key, eval_key, tell_key = jax.random.split(key, 4)
        
        population, es_state = jit_ask(ask_key, es_state, es_params)
        fitness = scoring_fn(population, eval_key)
        # evosax minimizes, so negate fitness for maximization
        es_state, _ = jit_tell(tell_key, population, -fitness, es_state, es_params)
        
        # Copy to host for numpy operations
        fitness_host = jax.device_get(fitness)
        population_host = jax.device_get(population)
        
        gen_best = float(np.max(fitness_host))
        gen_mean = float(np.mean(fitness_host))
        
        # ES optimizes the mean - that's the actual solution
        # Population is just noise samples for gradient estimation
        current_mean = jax.device_get(es_state.mean)
        
        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best
        
        wandb.log({
            "generation": gen, "best_fitness": gen_best,
            "mean_fitness": gen_mean, "best_overall": best_overall_fitness,
        })
        
        if gen % args.log_interval == 0 or gen == num_generations - 1:
            print(f"Gen {gen:4d} | Best: {gen_best:8.2f} | Mean: {gen_mean:8.2f} | Overall: {best_overall_fitness:8.2f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Best overall: {best_overall_fitness:.2f}")
    
    # Use ES mean as the best solution (ES optimizes the mean, not individual samples)
    best_params = jax.device_get(es_state.mean)
    
    # Evaluate the mean to get actual fitness
    key, eval_key = jax.random.split(key)
    best_fitness = float(scoring_fn(best_params[None, :], eval_key)[0])
    
    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"es_{env_name}_leg{leg_label}_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': np.array(best_params),
            'param_template': param_template,
            'best_fitness': best_fitness,
            'damaged_leg': damaged_leg,
            'config': config,
        }, f)
    print(f"Saved: {ckpt_path}")
    
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
    
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
