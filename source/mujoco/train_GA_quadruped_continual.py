"""
Train SimpleGA on Go1 with continual learning - leg damage experiment.

First task has no damage (healthy robot baseline).
Subsequent tasks damage one leg (locks joints in fixed position).
When switching tasks, a different random leg is chosen for damage.
Task switching: every 50 generations

Go1 action structure (12 actions total):
  FR (Front Right): indices 0-2 (hip, thigh, calf)
  FL (Front Left): indices 3-5 (hip, thigh, calf)
  RR (Rear Right): indices 6-8 (hip, thigh, calf)
  RL (Rear Left): indices 9-11 (hip, thigh, calf)

Usage:
    python train_simplega_go1_legdamage_continual.py --gpus 0
    python train_simplega_go1_legdamage_continual.py --num_tasks 20 --gens_per_task 50
"""

import argparse
import os
import sys

# Add repo root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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
from evosax.algorithms import SimpleGA
from mujoco_playground import registry
import time
import pickle
import wandb
import json
import numpy as np


# Leg action indices (None = healthy/no damage)
LEG_ACTION_INDICES = {
    None: [],        # Healthy (no damage)
    0: [0, 1, 2],    # Front Right (FR) - action indices
    1: [3, 4, 5],    # Front Left (FL) - action indices
    2: [6, 7, 8],    # Rear Right (RR) - action indices
    3: [9, 10, 11],  # Rear Left (RL) - action indices
}

# Leg qpos indices (joint positions in state.data.qpos)
# qpos layout: [0:7] = base pos/quat, [7:19] = 12 joint angles
LEG_QPOS_INDICES = {
    None: [],        # Healthy (no damage)
    0: [7, 8, 9],    # Front Right (FR) - qpos indices
    1: [10, 11, 12], # Front Left (FL) - qpos indices
    2: [13, 14, 15], # Rear Right (RR) - qpos indices
    3: [16, 17, 18], # Rear Left (RL) - qpos indices
}

# Leg qvel indices (joint velocities in state.data.qvel)
# qvel layout: [0:6] = base lin/ang vel, [6:18] = 12 joint velocities
LEG_QVEL_INDICES = {
    None: [],        # Healthy (no damage)
    0: [6, 7, 8],    # Front Right (FR) - qvel indices
    1: [9, 10, 11],  # Front Left (FL) - qvel indices
    2: [12, 13, 14], # Rear Right (RR) - qvel indices
    3: [15, 16, 17], # Rear Left (RL) - qvel indices
}

# Locked joint positions (bent/tucked position)
LOCKED_JOINT_POSITIONS = jnp.array([0.0, 1.2, -2.4])  # hip=0, thigh bent up, calf bent back

LEG_NAMES = {None: 'HEALTHY', 0: 'FR', 1: 'FL', 2: 'RR', 3: 'RL'}


class LegDamageWrapper:
    """Wrapper that locks a damaged leg in a fixed bent position.
    
    The damaged leg's joints are locked at a fixed position with zero velocity,
    simulating a completely broken/frozen limb that cannot move.
    """
    
    def __init__(self, env, damaged_leg_idx):
        """
        Args:
            env: The base environment
            damaged_leg_idx: Which leg to damage (0=FR, 1=FL, 2=RR, 3=RL), or None for healthy
        """
        self._env = env
        self._damaged_leg = damaged_leg_idx
        
        # Create action mask: 1 for healthy, 0 for damaged
        self._action_mask = jnp.ones(env.action_size)
        if damaged_leg_idx is not None:
            action_indices = jnp.array(LEG_ACTION_INDICES[damaged_leg_idx])
            self._action_mask = self._action_mask.at[action_indices].set(0.0)
            
            # Store qpos/qvel indices for locking
            self._qpos_indices = jnp.array(LEG_QPOS_INDICES[damaged_leg_idx])
            self._qvel_indices = jnp.array(LEG_QVEL_INDICES[damaged_leg_idx])
        else:
            self._qpos_indices = None
            self._qvel_indices = None
    
    def __getattr__(self, name):
        """Forward all other attribute access to the wrapped environment."""
        return getattr(self._env, name)
    
    def _lock_leg_joints(self, state):
        """Lock the damaged leg's joints to fixed position with zero velocity."""
        if self._damaged_leg is None:
            return state
        
        # Lock joint positions to bent position
        new_qpos = state.data.qpos.at[self._qpos_indices].set(LOCKED_JOINT_POSITIONS)
        # Lock joint velocities to zero
        new_qvel = state.data.qvel.at[self._qvel_indices].set(0.0)
        
        # Update state with locked joints
        new_data = state.data.replace(qpos=new_qpos, qvel=new_qvel)
        return state.replace(data=new_data)
    
    def step(self, state, action):
        # Zero out the damaged leg's actions
        masked_action = action * self._action_mask
        # Take step
        next_state = self._env.step(state, masked_action)
        # Lock the damaged leg's joints after the step
        next_state = self._lock_leg_joints(next_state)
        return next_state
    
    def reset(self, rng):
        state = self._env.reset(rng)
        # Lock the damaged leg's joints on reset too
        return self._lock_leg_joints(state)


def generate_random_leg_sequence(rng_key, num_tasks: int, avoid_consecutive: bool = True):
    """Generate a random sequence of legs to damage.
    
    All tasks damage a random leg.
    
    Args:
        rng_key: JAX random key
        num_tasks: Number of tasks
        avoid_consecutive: If True, avoid damaging same leg twice in a row
    
    Returns:
        List of leg indices (0-3 for damaged legs)
    """
    sequence = []
    
    for i in range(num_tasks):
        rng_key, subkey = random.split(rng_key)
        
        if avoid_consecutive and len(sequence) > 0:
            last_leg = sequence[-1]
            # Choose from legs other than the last one
            available_legs = [leg for leg in range(4) if leg != last_leg]
            idx = int(random.randint(subkey, (), 0, len(available_legs)))
            leg = available_legs[idx]
        else:
            # First task or no restriction
            leg = int(random.randint(subkey, (), 0, 4))
        
        sequence.append(leg)
    
    return sequence


# Neural network policy for locomotion
class MLPPolicy(nn.Module):
    """MLP policy network for locomotion environments.
    
    Uses SiLU/Swish activation to match Brax PPO networks.
    """
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
        done = next_state.done
        
        total_reward = total_reward + reward * (1.0 - done_flag)
        done_flag = jnp.maximum(done_flag, done)
        
        return (next_state, total_reward, done_flag, key), None
    
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)
    
    (_, total_reward, _, _), _ = jax.lax.scan(
        step_fn, (state, 0.0, 0.0, key), None, length=max_steps
    )
    
    return total_reward


def evaluate_params(flat_params, param_template, env, policy, key, max_steps=1000, obs_key='state'):
    params = unflatten_params(flat_params, param_template)
    return rollout_episode(env, policy, params, key, max_steps, obs_key)


def evaluate_population(flat_params_pop, param_template, env, policy, key, max_steps=1000, num_evals=1, obs_key='state'):
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
    parser = argparse.ArgumentParser(description='SimpleGA Continual Learning on Go1 with Leg Damage')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--env', type=str, default='Go1JoystickFlatTerrain',
                        help='Base environment name')
    parser.add_argument('--num_tasks', type=int, default=20,
                        help='Number of task switches (first is healthy, rest have leg damage)')
    parser.add_argument('--gens_per_task', type=int, default=50,
                        help='Generations per task')
    parser.add_argument('--pop_size', type=int, default=512)
    parser.add_argument('--mutation_std', type=float, default=0.1)
    parser.add_argument('--crossover_rate', type=float, default=0.0)
    parser.add_argument('--elite_ratio', type=float, default=0.1)
    parser.add_argument('--num_evals', type=int, default=3)
    parser.add_argument('--init_range', type=float, default=0.1)
    parser.add_argument('--hidden_dims', type=str, default='512,256,128')
    parser.add_argument('--wandb_project', type=str, default='mujoco_evosax')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--avoid_consecutive', action='store_true', default=True,
                        help='Avoid damaging same leg twice in a row')
    parser.add_argument('--trial', type=int, default=1,
                        help='Trial number for multiple runs')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configuration
    seed = args.seed
    pop_size = args.pop_size
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(','))
    max_episode_steps = 1000
    log_interval = 10
    total_generations = args.num_tasks * args.gens_per_task
    
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
        output_dir = os.path.join("projects", f"ga_go1_legdamage_seed{seed}_trial{args.trial}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize key
    key = jax.random.key(seed)
    
    # Generate leg damage sequence
    key, seq_key = jax.random.split(key)
    leg_sequence = generate_random_leg_sequence(seq_key, args.num_tasks, args.avoid_consecutive)
    
    print(f"\n{'='*60}")
    print(f"SimpleGA Continual Learning: Go1 Leg Damage")
    print(f"{'='*60}")
    print(f"  Base Environment: {args.env}")
    print(f"  Tasks: {args.num_tasks} (all with leg damage)")
    print(f"  Generations per task: {args.gens_per_task}")
    print(f"  Total generations: {total_generations}")
    print(f"  Population: {pop_size}")
    print(f"  Devices: {num_devices}")
    print(f"  Seed: {seed}, Trial: {args.trial}")
    
    print(f"\nLeg Damage Sequence:")
    for i, leg_idx in enumerate(leg_sequence):
        if leg_idx is None:
            print(f"  Task {i+1}: HEALTHY (no damage)")
        else:
            print(f"  Task {i+1}: Damage {LEG_NAMES[leg_idx]} leg (action indices {LEG_ACTION_INDICES[leg_idx]})")
    print(f"{'='*60}\n")
    
    # Load first env to get dimensions (healthy)
    first_env = registry.load(args.env)
    first_env = LegDamageWrapper(first_env, None)  # Healthy
    key, reset_key = jax.random.split(key)
    state = first_env.reset(reset_key)
    
    obs_key = 'state'
    obs_dim = state.obs[obs_key].shape[-1]
    action_dim = first_env.action_size
    
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    
    # Create policy network
    key, init_key = jax.random.split(key)
    policy, param_template = create_policy_network(init_key, obs_dim, action_dim, hidden_dims)
    
    flat_params = get_flat_params(param_template)
    num_params = flat_params.shape[0]
    print(f"  Network: {hidden_dims}, {num_params} params")
    
    # Initialize wandb
    run_name = args.run_name or f"ga_go1_legdamage_seed{seed}_trial{args.trial}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "algorithm": "SimpleGA",
            "continual": True,
            "experiment": "leg_damage",
            "base_env": args.env,
            "num_tasks": args.num_tasks,
            "gens_per_task": args.gens_per_task,
            "total_generations": total_generations,
            "pop_size": pop_size,
            "mutation_std": args.mutation_std,
            "elite_ratio": args.elite_ratio,
            "num_evals": args.num_evals,
            "hidden_dims": hidden_dims,
            "seed": seed,
            "trial": args.trial,
            "leg_sequence": [LEG_NAMES[leg] for leg in leg_sequence],
            "avoid_consecutive": args.avoid_consecutive,
        }
    )
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump({
            "algorithm": "SimpleGA",
            "base_env": args.env,
            "num_tasks": args.num_tasks,
            "gens_per_task": args.gens_per_task,
            "total_generations": total_generations,
            "pop_size": pop_size,
            "seed": seed,
            "trial": args.trial,
            "leg_sequence": [LEG_NAMES[leg] for leg in leg_sequence],
            "leg_action_indices": {LEG_NAMES[k]: v for k, v in LEG_ACTION_INDICES.items()},
            "avoid_consecutive": args.avoid_consecutive,
        }, f, indent=2)
    
    # Initialize SimpleGA
    dummy_solution = jnp.zeros(num_params)
    key, pop_init_key = jax.random.split(key)
    init_population = jax.random.normal(pop_init_key, (pop_size, num_params)) * args.init_range
    init_fitness = jnp.full(pop_size, jnp.inf)
    
    std_schedule = lambda gen: args.mutation_std
    
    es = SimpleGA(
        population_size=pop_size,
        solution=dummy_solution,
        std_schedule=std_schedule,
    )
    es.elite_ratio = args.elite_ratio
    es_params = jax.device_put(
        es.default_params.replace(crossover_rate=args.crossover_rate),
        replicate_sharding
    )
    
    key, init_key = jax.random.split(key)
    es_state = jax.jit(es.init, out_shardings=replicate_sharding)(
        init_key, init_population, init_fitness, es_params
    )
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_fitness_overall = -float('inf')
    best_params = None
    task_best_params = None  # Track best params for current task
    all_metrics = []
    start_time = time.time()
    global_gen = 0
    
    for task_idx, damaged_leg in enumerate(leg_sequence):
        leg_name = LEG_NAMES[damaged_leg]
        print(f"\n{'='*60}")
        print(f"Task {task_idx + 1}/{args.num_tasks}: {leg_name}")
        print(f"{'='*60}")
        
        # Load environment with leg damage for this task
        base_env = registry.load(args.env)
        env = LegDamageWrapper(base_env, damaged_leg)
        
        # JIT compile evaluation for this environment
        @jax.jit
        def jit_evaluate(flat_params_pop, key):
            return evaluate_population(
                flat_params_pop, param_template, env, policy, key, 
                max_episode_steps, args.num_evals, obs_key
            )
        
        @jax.jit
        def jit_ask(key, state, params):
            population, new_state = es.ask(key, state, params)
            population = jax.device_put(population, parallel_sharding)
            return population, new_state
        
        @jax.jit
        def jit_tell(key, population, fitness, state, params):
            return es.tell(key, population, fitness, state, params)
        
        # Warmup JIT for this env
        key, warmup_key, ask_key = jax.random.split(key, 3)
        warmup_pop, _ = jit_ask(ask_key, es_state, es_params)
        _ = jit_evaluate(warmup_pop, warmup_key)
        
        task_best_fitness = -float('inf')
        task_best_params = None  # Reset for each task
        
        for gen_in_task in range(args.gens_per_task):
            key, key_ask, key_eval, key_tell = jax.random.split(key, 4)
            
            # Ask
            population, es_state = jit_ask(key_ask, es_state, es_params)
            
            # Evaluate
            fitness = jit_evaluate(population, key_eval)
            
            # Tell (negate for minimization)
            es_state, _ = jit_tell(key_tell, population, -fitness, es_state, es_params)
            
            # Track best
            fitness_gathered = jax.device_get(fitness)
            population_gathered = jax.device_get(population)
            
            gen_best_idx = jnp.argmax(fitness_gathered)
            gen_best_fitness = float(fitness_gathered[gen_best_idx])
            mean_fitness = float(jnp.mean(fitness_gathered))
            
            if gen_best_fitness > task_best_fitness:
                task_best_fitness = gen_best_fitness
                task_best_params = population_gathered[gen_best_idx]
            if gen_best_fitness > best_fitness_overall:
                best_fitness_overall = gen_best_fitness
                best_params = population_gathered[gen_best_idx]
            
            # Log
            wandb.log({
                "generation": global_gen,
                "task": task_idx,
                "task_name": leg_name,
                "damaged_leg": leg_name,
                "fitness/best": gen_best_fitness,
                "fitness/mean": mean_fitness,
                "fitness/best_task": task_best_fitness,
                "fitness/best_overall": best_fitness_overall,
            }, step=global_gen)
            
            all_metrics.append({
                "global_gen": global_gen,
                "task": task_idx,
                "damaged_leg": leg_name,
                "best_fitness": gen_best_fitness,
                "mean_fitness": mean_fitness,
            })
            
            if gen_in_task % log_interval == 0 or gen_in_task == args.gens_per_task - 1:
                elapsed = time.time() - start_time
                print(f"Task {task_idx+1} [{leg_name}] Gen {gen_in_task:3d}/{args.gens_per_task} (Global {global_gen:4d}) | "
                      f"Best: {gen_best_fitness:8.2f} | Mean: {mean_fitness:8.2f} | "
                      f"Overall Best: {best_fitness_overall:8.2f} | Time: {elapsed:6.1f}s")
            
            global_gen += 1
        
        print(f"Task {task_idx + 1} [{leg_name}] complete. Best: {task_best_fitness:.2f}")
        
        # Save checkpoint at end of task
        ckpt_path = os.path.join(checkpoint_dir, f'task_{task_idx:02d}_{leg_name}.pkl')
        with open(ckpt_path, 'wb') as f:
            pickle.dump({
                'flat_params': task_best_params,
                'param_template': param_template,
                'hidden_dims': hidden_dims,
                'task_idx': task_idx,
                'damaged_leg': leg_name,
                'damaged_leg_idx': damaged_leg,
                'task_best_fitness': task_best_fitness,
                'global_gen': global_gen - 1,  # Last generation of this task
            }, f)
        print(f"  Saved checkpoint: {ckpt_path}")
    
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n{'='*60}")
    print("Continual Learning Complete!")
    print(f"{'='*60}")
    print(f"  Total tasks: {args.num_tasks}")
    print(f"  Total generations: {global_gen}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Best fitness overall: {best_fitness_overall:.2f}")
    
    # Save checkpoint
    ckpt_path = os.path.join(output_dir, "ga_go1_legdamage_best_params.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': best_params,
            'param_template': param_template,
            'hidden_dims': hidden_dims,
            'best_fitness': best_fitness_overall,
            'leg_sequence': leg_sequence,
        }, f)
    
    # Save metrics
    with open(os.path.join(output_dir, "training_curve.json"), 'w') as f:
        json.dump({
            "metrics": all_metrics,
            "best_fitness": best_fitness_overall,
            "total_time": total_time,
            "leg_sequence": [LEG_NAMES[leg] for leg in leg_sequence],
        }, f, indent=2)
    
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
