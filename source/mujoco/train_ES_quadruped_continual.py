"""
Train OpenES on Go1 with continual learning - leg damage experiment.

Each task damages a different random leg (locks joints in fixed position).
Task switching: every N generations

Go1 action structure (12 actions total):
  FR (Front Right): indices 0-2 (hip, thigh, calf)
  FL (Front Left): indices 3-5 (hip, thigh, calf)
  RR (Rear Right): indices 6-8 (hip, thigh, calf)
  RL (Rear Left): indices 9-11 (hip, thigh, calf)

Usage:
    python train_openes_go1_legdamage_continual.py --gpus 0
    python train_openes_go1_legdamage_continual.py --num_tasks 20 --gens_per_task 50
"""

import argparse
import os
import sys

# Parse --gpus argument BEFORE importing JAX
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
from jax import random
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import flax.linen as nn
from evosax.algorithms import Open_ES
import optax
from mujoco_playground import registry
import time
import pickle
import wandb
import json
import numpy as np
import imageio


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
    """Generate a random sequence of legs to damage.
    
    All tasks damage a random leg.
    """
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


def rollout_episode(env, policy, params, key, max_steps=1000, obs_key='state'):
    def step_fn(carry, _):
        state, total_reward, done_flag, key = carry
        key, _ = jax.random.split(key)
        
        obs = state.obs[obs_key] if isinstance(state.obs, dict) else state.obs
        action = policy.apply(params, obs)
        next_state = env.step(state, action)
        reward = next_state.reward
        
        total_reward = total_reward + reward * (1.0 - done_flag)
        done_flag = jnp.maximum(done_flag, next_state.done)
        
        return (next_state, total_reward, done_flag, key), None
    
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)
    
    (_, total_reward, _, _), _ = jax.lax.scan(
        step_fn, (state, 0.0, 0.0, key), None, length=max_steps
    )
    
    return total_reward


def evaluate_params(flat_params, param_template, env, policy, key, max_steps, obs_key):
    params = unflatten_params(flat_params, param_template)
    return rollout_episode(env, policy, params, key, max_steps, obs_key)


def evaluate_population(flat_params_pop, param_template, env, policy, key, max_steps, num_evals, obs_key):
    pop_size = flat_params_pop.shape[0]
    
    if num_evals == 1:
        keys = jax.random.split(key, pop_size)
        fitness = jax.vmap(
            lambda fp, k: evaluate_params(fp, param_template, env, policy, k, max_steps, obs_key)
        )(flat_params_pop, keys)
    else:
        all_keys = jax.random.split(key, pop_size * num_evals)
        flat_params_repeated = jnp.repeat(flat_params_pop, num_evals, axis=0)
        
        all_rewards = jax.vmap(
            lambda fp, k: evaluate_params(fp, param_template, env, policy, k, max_steps, obs_key)
        )(flat_params_repeated, all_keys)
        
        all_rewards = all_rewards.reshape(pop_size, num_evals)
        fitness = jnp.mean(all_rewards, axis=1)
    
    return fitness


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Go1JoystickFlatTerrain')
    parser.add_argument('--num_tasks', type=int, default=20)
    parser.add_argument('--gens_per_task', type=int, default=50)
    parser.add_argument('--pop_size', type=int, default=512)
    parser.add_argument('--sigma', type=float, default=0.04)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_evals', type=int, default=1)
    parser.add_argument('--init_range', type=float, default=0.1)
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 128])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--avoid_consecutive', action='store_true', default=True)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='openes_go1_legdamage_continual')
    parser.add_argument('--gpus', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    
    env_name = args.env
    num_tasks = args.num_tasks
    gens_per_task = args.gens_per_task
    total_generations = num_tasks * gens_per_task
    pop_size = args.pop_size
    sigma = args.sigma
    learning_rate = args.learning_rate
    num_evals = args.num_evals
    max_episode_steps = args.max_episode_steps
    hidden_dims = tuple(args.hidden_dims)
    seed = args.seed
    log_interval = args.log_interval
    
    # Setup sharding
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(devices, ("devices",))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())
    parallel_sharding = NamedSharding(mesh, PartitionSpec("devices"))
    
    if pop_size % num_devices != 0:
        pop_size = (pop_size // num_devices + 1) * num_devices
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("projects", f"openes_go1_legdamage_continual_trial{args.trial}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"OpenES Go1 Leg Damage Continual Learning")
    print(f"{'='*60}")
    print(f"  Tasks: {num_tasks}")
    print(f"  Generations per task: {gens_per_task}")
    print(f"  Total generations: {total_generations}")
    print(f"  Population: {pop_size}")
    print(f"  Devices: {num_devices}")
    print(f"  Avoid consecutive same leg: {args.avoid_consecutive}")
    
    # Initialize key
    key = jax.random.key(seed)
    
    # Generate leg damage sequence
    key, seq_key = jax.random.split(key)
    leg_sequence = generate_random_leg_sequence(seq_key, num_tasks, args.avoid_consecutive)
    print(f"\nLeg damage sequence:")
    for i, leg in enumerate(leg_sequence):
        print(f"  Task {i}: {LEG_NAMES[leg]}")
    
    # Load base environment
    base_env = registry.load(env_name)
    key, reset_key = jax.random.split(key)
    state = base_env.reset(reset_key)
    
    obs_key = 'state'
    obs_dim = state.obs[obs_key].shape[-1]
    action_dim = base_env.action_size
    
    print(f"\n  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    
    # Create policy network
    key, init_key = jax.random.split(key)
    policy, param_template = create_policy_network(init_key, obs_dim, action_dim, hidden_dims)
    
    flat_params = get_flat_params(param_template)
    num_params = flat_params.shape[0]
    print(f"  Network: {hidden_dims}, {num_params} params")
    
    # Initialize wandb
    run_name = args.run_name or args.experiment_name or f"openes_go1_legdamage_continual_trial{args.trial}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "algorithm": "OpenES",
            "continual": True,
            "env_name": env_name,
            "num_tasks": num_tasks,
            "gens_per_task": gens_per_task,
            "total_generations": total_generations,
            "pop_size": pop_size,
            "sigma": sigma,
            "learning_rate": learning_rate,
            "num_evals": num_evals,
            "hidden_dims": hidden_dims,
            "seed": seed,
            "trial": args.trial,
            "leg_sequence": [LEG_NAMES[l] for l in leg_sequence],
        }
    )
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump({
            "algorithm": "OpenES",
            "continual": True,
            "env_name": env_name,
            "num_tasks": num_tasks,
            "gens_per_task": gens_per_task,
            "total_generations": total_generations,
            "pop_size": pop_size,
            "sigma": sigma,
            "learning_rate": learning_rate,
            "num_evals": num_evals,
            "hidden_dims": hidden_dims,
            "seed": seed,
            "trial": args.trial,
            "leg_sequence": [LEG_NAMES[l] for l in leg_sequence],
        }, f, indent=2)
    
    # Initialize OpenES
    print("\nInitializing OpenES...")
    dummy_solution = jnp.zeros(num_params)
    std_schedule = lambda gen: sigma
    optimizer = optax.adam(learning_rate=learning_rate)
    
    es = Open_ES(
        population_size=pop_size, 
        solution=dummy_solution,
        std_schedule=std_schedule,
        optimizer=optimizer,
        use_antithetic_sampling=True,
    )
    es_params = jax.device_put(es.default_params, replicate_sharding)
    
    key, init_key = jax.random.split(key)
    # API: es.init(key, mean, params)
    es_state = es.init(init_key, dummy_solution, es_params)
    es_state = jax.device_put(es_state, replicate_sharding)
    
    print(f"  Population size: {pop_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Sigma: {sigma}")
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting continual training...")
    print(f"{'='*60}\n")
    
    best_fitness_overall = -float('inf')
    best_params = None
    best_fitness_per_task = [-float('inf')] * num_tasks
    task_best_params = None  # Track best params for current task
    training_metrics_list = []
    
    start_time = time.time()
    global_gen = 0
    
    for task_num, damaged_leg in enumerate(leg_sequence):
        leg_name = LEG_NAMES[damaged_leg]
        print(f"\n{'='*40}")
        print(f"Task {task_num}: Damaged Leg = {leg_name}")
        print(f"{'='*40}")
        
        # Create wrapped environment for this task
        wrapped_env = LegDamageWrapper(base_env, damaged_leg)
        
        # Create JIT-compiled functions for this environment
        @jax.jit
        def jit_evaluate(flat_params_pop, key):
            return evaluate_population(
                flat_params_pop, param_template, wrapped_env, policy, key, 
                max_episode_steps, num_evals, obs_key
            )
        
        @jax.jit
        def jit_ask(key, state, params):
            population, new_state = es.ask(key, state, params)
            population = jax.device_put(population, parallel_sharding)
            return population, new_state
        
        @jax.jit
        def jit_tell(key, population, fitness, state, params):
            return es.tell(key, population, fitness, state, params)
        
        # Warmup JIT for this task
        key, warmup_key, ask_key = jax.random.split(key, 3)
        warmup_pop, _ = jit_ask(ask_key, es_state, es_params)
        _ = jit_evaluate(warmup_pop, warmup_key)
        
        task_best_fitness = -float('inf')
        task_best_params = None  # Reset for each task
        
        for gen_in_task in range(gens_per_task):
            key, key_ask, key_eval, key_tell = jax.random.split(key, 4)
            
            population, es_state = jit_ask(key_ask, es_state, es_params)
            fitness = jit_evaluate(population, key_eval)
            es_state, _ = jit_tell(key_tell, population, -fitness, es_state, es_params)
            
            # Track best
            fitness_gathered = jax.device_get(fitness)
            population_gathered = jax.device_get(population)
            
            gen_best_idx = jnp.argmax(fitness_gathered)
            gen_best_fitness = float(fitness_gathered[gen_best_idx])
            
            if gen_best_fitness > task_best_fitness:
                task_best_fitness = gen_best_fitness
                best_fitness_per_task[task_num] = task_best_fitness
                task_best_params = population_gathered[gen_best_idx]
            
            if gen_best_fitness > best_fitness_overall:
                best_fitness_overall = gen_best_fitness
                best_params = population_gathered[gen_best_idx]
            
            mean_fitness = float(jnp.mean(fitness_gathered))
            std_fitness = float(jnp.std(fitness_gathered))
            
            wandb.log({
                "generation": global_gen,
                "task": task_num,
                "damaged_leg": leg_name,
                "fitness/best": gen_best_fitness,
                "fitness/mean": mean_fitness,
                "fitness/std": std_fitness,
                "fitness/best_task": task_best_fitness,
                "fitness/best_overall": best_fitness_overall,
            }, step=global_gen)
            
            elapsed = time.time() - start_time
            training_metrics_list.append({
                'generation': global_gen,
                'task': task_num,
                'damaged_leg': leg_name,
                'best_fitness': gen_best_fitness,
                'mean_fitness': mean_fitness,
                'task_best': task_best_fitness,
                'overall_best': best_fitness_overall,
                'elapsed_time': elapsed,
            })
            
            if gen_in_task % log_interval == 0 or gen_in_task == gens_per_task - 1:
                print(f"  Gen {global_gen:4d} (Task {task_num}, {leg_name}) | "
                      f"Best: {gen_best_fitness:8.2f} | "
                      f"Mean: {mean_fitness:8.2f} | "
                      f"Task Best: {task_best_fitness:8.2f} | "
                      f"Overall: {best_fitness_overall:8.2f}")
            
            global_gen += 1
        
        print(f"\nTask {task_num} ({leg_name}) complete. Best: {task_best_fitness:.2f}")
        
        # Save checkpoint at end of task
        ckpt_path = os.path.join(checkpoint_dir, f'task_{task_num:02d}_{leg_name}.pkl')
        with open(ckpt_path, 'wb') as f:
            pickle.dump({
                'flat_params': task_best_params,
                'param_template': param_template,
                'hidden_dims': hidden_dims,
                'task_idx': task_num,
                'damaged_leg': leg_name,
                'damaged_leg_idx': damaged_leg,
                'task_best_fitness': task_best_fitness,
                'global_gen': global_gen - 1,  # Last generation of this task
            }, f)
        print(f"  Saved checkpoint: {ckpt_path}")
        
        # Save GIFs at end of each task
        try:
            task_gifs_dir = os.path.join(output_dir, "gifs", f"task_{task_num:02d}_{leg_name}")
            os.makedirs(task_gifs_dir, exist_ok=True)
            
            params = unflatten_params(task_best_params, param_template)
            jit_policy = jax.jit(policy.apply)
            
            num_gifs = 3
            for gif_idx in range(num_gifs):
                key, gif_key = jax.random.split(key)
                jit_reset = jax.jit(env.reset)
                jit_step = jax.jit(env.step)
                
                state = jit_reset(gif_key)
                trajectory = [state]
                total_reward = 0.0
                
                for _ in range(max_episode_steps):
                    obs = jnp.expand_dims(state.obs, axis=0)
                    action = jit_policy(params, obs)[0]
                    # Apply leg damage mask
                    if damaged_leg is not None:
                        damaged_indices = jnp.array(LEG_ACTION_INDICES[damaged_leg])
                        action = action.at[damaged_indices].set(0.0)
                    state = jit_step(state, action)
                    trajectory.append(state)
                    total_reward += float(state.reward)
                    if state.done:
                        break
                
                # Render every 2nd frame
                images = env.render(trajectory[::2], height=240, width=320, camera="tracking")
                gif_path = os.path.join(task_gifs_dir, f"trial{gif_idx}_reward{total_reward:.0f}.gif")
                imageio.mimsave(gif_path, images, fps=30, loop=0)
            
            print(f"  Saved {num_gifs} GIFs to: {task_gifs_dir}")
        except Exception as e:
            print(f"  Warning: Failed to save GIFs for task {task_num}: {e}")
    
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n{'='*60}")
    print("Continual Training Complete!")
    print(f"{'='*60}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Best overall fitness: {best_fitness_overall:.2f}")
    print(f"\n  Best per task:")
    for i, (leg, best) in enumerate(zip(leg_sequence, best_fitness_per_task)):
        print(f"    Task {i} ({LEG_NAMES[leg]}): {best:.2f}")
    
    # Save best params
    ckpt_path = os.path.join(output_dir, "openes_go1_legdamage_continual_best_params.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': best_params,
            'param_template': param_template,
            'hidden_dims': hidden_dims,
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'best_fitness': best_fitness_overall,
            'best_per_task': best_fitness_per_task,
            'leg_sequence': [LEG_NAMES[l] for l in leg_sequence],
        }, f)
    print(f"  Saved checkpoint: {ckpt_path}")
    
    # Save training curve
    curve_path = os.path.join(output_dir, "training_curve.json")
    with open(curve_path, 'w') as f:
        json.dump({
            "metrics": training_metrics_list,
            "best_fitness": best_fitness_overall,
            "best_per_task": best_fitness_per_task,
            "total_time": total_time,
            "leg_sequence": [LEG_NAMES[l] for l in leg_sequence],
        }, f, indent=2)
    
    wandb.finish()
    print(f"\nDone!")


if __name__ == "__main__":
    main()
