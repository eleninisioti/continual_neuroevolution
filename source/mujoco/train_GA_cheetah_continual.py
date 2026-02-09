"""
Train SimpleGA on MuJoCo Playground with Continual Learning (physics changes).

This script trains across multiple tasks where physics (gravity or friction) is modified between tasks.
The ES state is preserved across tasks (no reset), enabling continual learning.

Usage:
    python train_simplega_continual.py --env HumanoidWalk
    python train_simplega_continual.py --env CheetahRun --num_tasks 5 --task_mod gravity
    python train_simplega_continual.py --env WalkerRun --task_mod friction --friction_low_mult 0.2 --friction_high_mult 5.0
"""

import argparse
import os
import sys

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
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import flax.linen as nn
from evosax.algorithms import SimpleGA
from mujoco_playground import registry
import time
import pickle
import wandb
import json
import numpy as np
import matplotlib.pyplot as plt
import imageio


# Neural network policy
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
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    return flat_params


def unflatten_params(flat_params, param_template):
    """Unflatten a vector back to nested parameter dict."""
    _, unravel_fn = jax.flatten_util.ravel_pytree(param_template)
    return unravel_fn(flat_params)


def create_policy_network(key, obs_dim, action_dim, hidden_dims=(128, 128)):
    """Initialize the policy network and return params."""
    policy = MLPPolicy(hidden_dims=hidden_dims, action_dim=action_dim)
    dummy_obs = jnp.zeros((obs_dim,))
    params = policy.init(key, dummy_obs)
    return policy, params


def rollout_episode(env, policy, params, key, max_steps=1000):
    """Rollout a single episode and return total reward."""
    
    def step_fn(carry, _):
        state, total_reward, done_flag, key = carry
        key, action_key = jax.random.split(key)
        
        # Get action from policy
        obs = state.obs
        action = policy.apply(params, obs)
        
        # Step environment
        next_state = env.step(state, action)
        reward = next_state.reward
        
        # Check if done
        done = next_state.done
        
        # Accumulate reward (only if not done)
        total_reward = total_reward + reward * (1.0 - done_flag)
        done_flag = jnp.maximum(done_flag, done)
        
        return (next_state, total_reward, done_flag, key), None
    
    # Reset environment
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)
    
    # Run episode
    (final_state, total_reward, _, _), _ = jax.lax.scan(
        step_fn, (state, 0.0, 0.0, key), None, length=max_steps
    )
    
    return total_reward


def evaluate_params(flat_params, param_template, env, policy, key, max_steps=1000):
    """Evaluate a single set of parameters."""
    params = unflatten_params(flat_params, param_template)
    return rollout_episode(env, policy, params, key, max_steps)


def evaluate_population(flat_params_pop, param_template, env, policy, key, max_steps=1000, num_evals=1):
    """Evaluate entire population in parallel using vmap.
    
    If num_evals > 1, each agent is evaluated multiple times and fitness is averaged.
    All evaluations are parallelized using vmap.
    """
    pop_size = flat_params_pop.shape[0]
    
    if num_evals == 1:
        # Single evaluation per agent
        keys = jax.random.split(key, pop_size)
        fitness = jax.vmap(
            lambda fp, k: evaluate_params(fp, param_template, env, policy, k, max_steps)
        )(flat_params_pop, keys)
    else:
        # Multiple evaluations per agent, averaged
        # Generate keys for all (pop_size * num_evals) evaluations
        all_keys = jax.random.split(key, pop_size * num_evals)
        
        # Tile params so each agent is repeated num_evals times
        flat_params_repeated = jnp.repeat(flat_params_pop, num_evals, axis=0)
        
        all_rewards = jax.vmap(
            lambda fp, k: evaluate_params(fp, param_template, env, policy, k, max_steps)
        )(flat_params_repeated, all_keys)
        
        # Reshape to (pop_size, num_evals) and take mean
        all_rewards = all_rewards.reshape(pop_size, num_evals)
        fitness = jnp.mean(all_rewards, axis=1)
    
    return fitness


def rollout_episode_with_trajectory(env, policy, params, key, max_steps=1000):
    """Rollout a single episode and return total reward AND trajectory for rendering."""
    
    def step_fn(carry, _):
        state, total_reward, done_flag, key = carry
        key, action_key = jax.random.split(key)
        
        # Get action from policy
        obs = state.obs
        action = policy.apply(params, obs)
        
        # Step environment
        next_state = env.step(state, action)
        reward = next_state.reward
        
        # Check if done
        done = next_state.done
        
        # Accumulate reward (only if not done)
        total_reward = total_reward + reward * (1.0 - done_flag)
        done_flag = jnp.maximum(done_flag, done)
        
        return (next_state, total_reward, done_flag, key), next_state
    
    # Reset environment
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)
    
    # Run episode and collect trajectory
    (final_state, total_reward, _, _), trajectory = jax.lax.scan(
        step_fn, (state, 0.0, 0.0, key), None, length=max_steps
    )
    
    return total_reward, trajectory


def final_evaluation(env, policy, best_params, param_template, key, env_name, multiplier,
                     num_eval_trials=100, max_steps=1000, save_dir="eval_results"):
    """Run final evaluation with GIF saving, JSON rewards, and histogram."""
    
    print("\n" + "=" * 60)
    print(f"Running Final Evaluation ({num_eval_trials} trials, multiplier={multiplier:.2f})...")
    print("=" * 60)
    
    # Create save directory
    import os
    os.makedirs(save_dir, exist_ok=True)
    gifs_dir = os.path.join(save_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)
    
    # Unflatten best params
    params = unflatten_params(best_params, param_template)
    
    rewards = []
    
    for trial in range(num_eval_trials):
        key, eval_key = jax.random.split(key)
        
        # Run episode and get trajectory
        reward, trajectory = rollout_episode_with_trajectory(
            env, policy, params, eval_key, max_steps
        )
        reward_val = float(reward)
        rewards.append(reward_val)
        
        # Convert trajectory to list of states for rendering
        trajectory_list = []
        for t in range(0, max_steps, 5):  # Sample every 5th frame to reduce GIF size
            state_t = jax.tree.map(lambda x: x[t], trajectory)
            trajectory_list.append(state_t)
        
        # Render trajectory to images
        try:
            images = env.render(trajectory_list, height=240, width=320, camera="side")
            
            # Save as GIF
            gif_path = os.path.join(gifs_dir, f"trial_{trial:03d}_reward_{reward_val:.1f}.gif")
            imageio.mimsave(gif_path, images, fps=30, loop=0)
            
            if trial % 10 == 0:
                print(f"  Trial {trial:3d}/{num_eval_trials}: reward = {reward_val:.2f}")
        except Exception as e:
            print(f"  Trial {trial:3d}: reward = {reward_val:.2f} (GIF save failed: {e})")
    
    # Save rewards to JSON
    rewards_data = {
        "env_name": env_name,
        "multiplier": multiplier,
        "num_trials": num_eval_trials,
        "rewards": rewards,
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
    }
    
    json_path = os.path.join(save_dir, "eval_rewards.json")
    with open(json_path, 'w') as f:
        json.dump(rewards_data, f, indent=2)
    print(f"\n  Rewards saved to: {json_path}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(rewards):.2f}')
    plt.axvline(np.median(rewards), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(rewards):.2f}')
    plt.xlabel('Reward', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Final Evaluation Rewards - {env_name} (Continual)\n({num_eval_trials} trials, multiplier={multiplier:.2f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    hist_path = os.path.join(save_dir, "reward_histogram.png")
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Histogram saved to: {hist_path}")
    
    # Log to wandb
    wandb.log({
        "eval/mean_reward": float(np.mean(rewards)),
        "eval/std_reward": float(np.std(rewards)),
        "eval/min_reward": float(np.min(rewards)),
        "eval/max_reward": float(np.max(rewards)),
    })
    wandb.log({"eval/histogram": wandb.Image(hist_path)})
    
    # Log a few sample GIFs to wandb
    sample_gifs = sorted(os.listdir(gifs_dir))[:5]  # First 5 GIFs
    for gif_name in sample_gifs:
        gif_path = os.path.join(gifs_dir, gif_name)
        wandb.log({f"eval/gif_{gif_name}": wandb.Video(gif_path, fps=30, format="gif")})
    
    print(f"\n  Final Evaluation Results:")
    print(f"    Mean reward:   {np.mean(rewards):.2f}")
    print(f"    Std reward:    {np.std(rewards):.2f}")
    print(f"    Min reward:    {np.min(rewards):.2f}")
    print(f"    Max reward:    {np.max(rewards):.2f}")
    
    return rewards


def create_modified_env(env_name, task_mod='gravity', multiplier=1.0):
    """Create environment with modified physics (gravity or friction).
    
    Args:
        env_name: Name of the environment
        task_mod: 'gravity' or 'friction'
        multiplier: Multiplier for the physics parameter (1.0 = normal)
    """
    from mujoco import mjx
    
    # Load the environment normally
    env = registry.load(env_name)
    
    if task_mod == 'gravity':
        # Modify gravity (default is -9.81)
        gravity_z = -9.81 * multiplier
        env.mj_model.opt.gravity[2] = gravity_z
        print(f"  Gravity set: multiplier={multiplier}, gravity_z={gravity_z:.2f}")
    elif task_mod == 'friction':
        # Modify friction for all geoms (scale sliding, torsional, rolling friction)
        env.mj_model.geom_friction[:] *= multiplier
        print(f"  Friction set: multiplier={multiplier}")
    else:
        raise ValueError(f"Unknown task_mod: {task_mod}. Use 'gravity' or 'friction'.")
    
    # Re-create mjx_model from modified mj_model
    env._mjx_model = mjx.put_model(env.mj_model)
    
    return env


def sample_task_multipliers(num_tasks, default_mult=1.0, low_mult=0.2, high_mult=5.0):
    """Generate multiplier values for each task, cycling through three values."""
    multiplier_cycle = [default_mult, low_mult, high_mult]
    multipliers = jnp.array([multiplier_cycle[i % 3] for i in range(num_tasks)])
    return multipliers


def parse_args():
    parser = argparse.ArgumentParser(description='SimpleGA Continual Learning on MuJoCo Playground')
    parser.add_argument('--env', type=str, default='CheetahRun',
                        help='Environment name (e.g., HumanoidWalk, CheetahRun)')
    parser.add_argument('--num_tasks', type=int, default=30,
                        help='Number of tasks (default 30 = 10 repetitions of 3 gravity values)')
    parser.add_argument('--gens_per_task', type=int, default=500,
                        help='Generations per task (default 100)')
    parser.add_argument('--pop_size', type=int, default=512,
                        help='Population size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    # Task modification type
    parser.add_argument('--task_mod', type=str, default='gravity', choices=['gravity', 'friction'],
                        help='Which physics parameter to modify: gravity or friction (default: gravity)')
    # Gravity settings (cycle: default -> low -> high)
    parser.add_argument('--gravity_default_mult', type=float, default=1.0,
                        help='Default gravity multiplier (default 1.0 = normal Earth gravity)')
    parser.add_argument('--gravity_low_mult', type=float, default=0.2,
                        help='Low gravity multiplier (default 0.2 = 1/5 Earth gravity)')
    parser.add_argument('--gravity_high_mult', type=float, default=5.0,
                        help='High gravity multiplier (default 5.0 = 5x Earth gravity)')
    # Friction settings (cycle: default -> low -> high)
    parser.add_argument('--friction_default_mult', type=float, default=1.0,
                        help='Default friction multiplier (default 1.0 = normal friction)')
    parser.add_argument('--friction_low_mult', type=float, default=0.2,
                        help='Low friction multiplier (default 0.2 = slippery/icy)')
    parser.add_argument('--friction_high_mult', type=float, default=5.0,
                        help='High friction multiplier (default 5.0 = sticky/rough)')
    parser.add_argument('--mutation_std', type=float, default=1.0,
                        help='Mutation standard deviation (default 1.0)')
    parser.add_argument('--crossover_rate', type=float, default=0.2,
                        help='Crossover rate (default 0.0, no crossover)')
    parser.add_argument('--elite_ratio', type=float, default=0.5,
                        help='Elite ratio - fraction of population used as parents (default 0.5)')
    parser.add_argument('--num_evals', type=int, default=1,
                        help='Number of evaluations per agent (averaged, default 1)')
    parser.add_argument('--init_range', type=float, default=1.0,
                        help='Initial population range: values drawn from N(0, init_range) (default 1.0)')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs to use (e.g., "0,3,4"). Default: all available')
    # Output and logging
    parser.add_argument('--wandb_project', type=str, default='mujoco_evosax',
                        help='Wandb project name (default: mujoco_evosax)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Wandb run name. If None, auto-generated from config.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for saving results. If None, uses default naming.')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configuration
    seed = args.seed
    pop_size = args.pop_size
    num_tasks = args.num_tasks
    gens_per_task = args.gens_per_task
    max_episode_steps = 1000
    hidden_dims = (32, 32, 32, 32)  # 4 layers of 32
    log_interval = 10
    env_name = args.env
    task_mod = args.task_mod
    gravity_default_mult = args.gravity_default_mult
    gravity_low_mult = args.gravity_low_mult
    gravity_high_mult = args.gravity_high_mult
    friction_default_mult = args.friction_default_mult
    friction_low_mult = args.friction_low_mult
    friction_high_mult = args.friction_high_mult
    mutation_std = args.mutation_std
    crossover_rate = args.crossover_rate
    elite_ratio = args.elite_ratio
    num_evals = args.num_evals
    init_range = args.init_range
    
    # Setup multi-GPU sharding
    # CUDA_VISIBLE_DEVICES is already set at script start based on --gpus
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(devices, ("devices",))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())
    parallel_sharding = NamedSharding(mesh, PartitionSpec("devices"))
    
    # Ensure pop_size is divisible by num_devices
    if pop_size % num_devices != 0:
        old_pop_size = pop_size
        pop_size = (pop_size // num_devices + 1) * num_devices
        print(f"Warning: Adjusted pop_size from {old_pop_size} to {pop_size} (divisible by {num_devices} GPUs)")
    
    print("=" * 60)
    print(f"SimpleGA Continual Learning on {env_name} (MuJoCo Playground/MJX)")
    print("=" * 60)
    print(f"\nUsing {num_devices} GPU(s): {[str(d) for d in devices]}")
    print(f"\nContinual Learning Setup:")
    print(f"  Task modification: {task_mod}")
    print(f"  Number of tasks: {num_tasks}")
    print(f"  Generations per task: {gens_per_task}")
    
    # Sample task multipliers based on task_mod
    key = jax.random.key(seed)
    if task_mod == 'gravity':
        multipliers = sample_task_multipliers(num_tasks, gravity_default_mult, gravity_low_mult, gravity_high_mult)
        print(f"  Gravity multipliers: {gravity_default_mult}x -> {gravity_low_mult}x -> {gravity_high_mult}x (cycling)")
    else:  # friction
        multipliers = sample_task_multipliers(num_tasks, friction_default_mult, friction_low_mult, friction_high_mult)
        print(f"  Friction multipliers: {friction_default_mult}x -> {friction_low_mult}x -> {friction_high_mult}x (cycling)")
    
    multiplier_list = [float(m) for m in multipliers]
    print(f"  Multiplier values: {[f'{m:.2f}' for m in multiplier_list]}")
    print(f"  Mutation std: {mutation_std}")
    print(f"  Crossover rate: {crossover_rate}")
    print(f"  Elite ratio: {elite_ratio}")
    
    # Determine output directory
    mod_suffix = "gravity" if task_mod == "gravity" else "friction"
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"results_ga_continual_{env_name.lower()}_{mod_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Initialize wandb
    run_name = args.run_name if args.run_name else f"ga_continual_{env_name}_{task_mod}_seed{seed}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "algorithm": "SimpleGA",
            "env_name": env_name,
            "task_mod": task_mod,
            "pop_size": pop_size,
            "num_tasks": num_tasks,
            "gens_per_task": gens_per_task,
            "total_generations": num_tasks * gens_per_task,
            "max_episode_steps": max_episode_steps,
            "hidden_dims": hidden_dims,
            "seed": seed,
            "mutation_std": mutation_std,
            "crossover_rate": crossover_rate,
            "elite_ratio": elite_ratio,
            "num_evals": num_evals,
            "init_range": init_range,
            "num_gpus": num_devices,
            "gpu_ids": [d.id for d in devices],
            "gravity_default_mult": gravity_default_mult,
            "gravity_low_mult": gravity_low_mult,
            "gravity_high_mult": gravity_high_mult,
            "friction_default_mult": friction_default_mult,
            "friction_low_mult": friction_low_mult,
            "friction_high_mult": friction_high_mult,
            "task_multipliers": multiplier_list,
            "continual": True,
            "output_dir": output_dir,
        }
    )
    
    # Save config to JSON file
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(wandb.config.as_dict(), f, indent=2)
    print(f"Config saved to: {config_path}")
    
    # Initialize training metrics list for CSV
    training_metrics_list = []
    
    # Create initial environment (with first multiplier)
    print(f"\nInitializing {env_name} environment...")
    initial_multiplier = multiplier_list[0]
    env = create_modified_env(env_name, task_mod, initial_multiplier)
    
    # Get observation and action dimensions
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)
    obs_dim = state.obs.shape[-1]
    action_dim = env.action_size
    
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Episode length: {max_episode_steps}")
    
    # Create policy network
    print("\nInitializing policy network...")
    key, init_key = jax.random.split(key)
    policy, param_template = create_policy_network(
        init_key, obs_dim, action_dim, hidden_dims
    )
    
    # Get number of parameters
    flat_params = get_flat_params(param_template)
    num_params = flat_params.shape[0]
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Total parameters: {num_params}")
    
    # Initialize SimpleGA with tuned hyperparameters
    print("\nInitializing SimpleGA...")
    dummy_solution = jnp.zeros(num_params)
    # Initialize population with random values from N(0, init_range)
    # This gives better initial diversity than all zeros
    key, pop_init_key = jax.random.split(key)
    init_population = jax.random.normal(pop_init_key, (pop_size, num_params)) * init_range
    init_fitness = jnp.full(pop_size, jnp.inf)  # Unknown fitness initially
    
    # Custom std schedule (constant at mutation_std)
    std_schedule = lambda gen: mutation_std
    
    es = SimpleGA(
        population_size=pop_size, 
        solution=dummy_solution,
        std_schedule=std_schedule,
    )
    # Override elite_ratio (hardcoded in class)
    es.elite_ratio = elite_ratio
    # Override crossover rate in params - replicate across devices
    es_params = jax.device_put(
        es.default_params.replace(crossover_rate=crossover_rate),
        replicate_sharding
    )
    
    key, init_key = jax.random.split(key)
    # Initialize ES state with sharding (replicated)
    es_state = jax.jit(es.init, out_shardings=replicate_sharding)(
        init_key, init_population, init_fitness, es_params
    )
    
    print(f"  Population size: {pop_size}")
    print(f"  Elite ratio: {elite_ratio} ({es.num_elites} elites)")
    print(f"  Mutation std: {mutation_std}")
    print(f"  Crossover rate: {crossover_rate}")
    print(f"  Num evals per agent: {num_evals}")
    print(f"  Init range: {init_range}")
    print(f"  Num GPUs: {num_devices}")
    
    # Training tracking
    best_fitness_overall = -float('inf')
    best_params_overall = None
    total_gen = 0
    start_time = time.time()
    
    # Continual learning loop - iterate through tasks
    for task_idx in range(num_tasks):
        # Reset task-best params for this task
        task_best_params = None
        current_multiplier = multiplier_list[task_idx]
        
        print("\n" + "=" * 60)
        print(f"TASK {task_idx + 1}/{num_tasks} - {task_mod}: {current_multiplier:.2f}")
        print("=" * 60)
        
        # Create environment with new multiplier
        env = create_modified_env(env_name, task_mod, current_multiplier)
        
        # JIT compile evaluation for this environment
        @jax.jit
        def jit_evaluate(flat_params_pop, key):
            return evaluate_population(
                flat_params_pop, param_template, env, policy, key, max_episode_steps, num_evals
            )
        
        # JIT compiled ask with sharding: population sharded, state replicated
        @jax.jit
        def jit_ask(key, state, params):
            population, new_state = es.ask(key, state, params)
            # Shard population across devices
            population = jax.device_put(population, parallel_sharding)
            return population, new_state
        
        # JIT compiled tell - gathers fitness from all devices
        @jax.jit
        def jit_tell(key, population, fitness, state, params):
            return es.tell(key, population, fitness, state, params)
        
        # Warmup JIT for new environment
        key, warmup_key, ask_key = jax.random.split(key, 3)
        warmup_pop, _ = jit_ask(ask_key, es_state, es_params)
        _ = jit_evaluate(warmup_pop, warmup_key)
        
        best_fitness_task = -float('inf')
        
        # Training loop for this task
        for gen in range(gens_per_task):
            # Split keys for this generation
            key, key_ask, key_eval, key_tell = jax.random.split(key, 4)
            
            # Ask for new population (sharded across GPUs)
            population, es_state = jit_ask(key_ask, es_state, es_params)
            
            # Evaluate population (parallel across GPUs)
            fitness = jit_evaluate(population, key_eval)
            
            # Tell strategy the fitness (evosax minimizes, so negate for maximization)
            es_state, _ = jit_tell(key_tell, population, -fitness, es_state, es_params)
            
            # Track best - need to gather from all devices
            fitness_gathered = jax.device_get(fitness)
            population_gathered = jax.device_get(population)
            
            gen_best_idx = jnp.argmax(fitness_gathered)
            gen_best_fitness = fitness_gathered[gen_best_idx]
            
            if gen_best_fitness > best_fitness_task:
                best_fitness_task = gen_best_fitness
                task_best_params = population_gathered[gen_best_idx]
            
            if gen_best_fitness > best_fitness_overall:
                best_fitness_overall = gen_best_fitness
                best_params_overall = population_gathered[gen_best_idx]
            
            # Logging
            mean_fitness = jnp.mean(fitness_gathered)
            std_fitness = jnp.std(fitness_gathered)
            
            # Log to wandb every generation (use generation as step for aligned x-axis with PPO)
            wandb.log({
                "generation": total_gen,
                "task": task_idx,
                "task_generation": gen,
                "multiplier": current_multiplier,
                "fitness/best": float(gen_best_fitness),
                "fitness/mean": float(mean_fitness),
                "fitness/std": float(std_fitness),
                "fitness/best_task": float(best_fitness_task),
                "fitness/best_overall": float(best_fitness_overall),
            }, step=total_gen)
            
            # Accumulate training metrics for CSV
            elapsed = time.time() - start_time
            training_metrics_list.append({
                'generation': total_gen,
                'task': task_idx,
                'task_generation': gen,
                'multiplier': current_multiplier,
                'best_fitness': float(gen_best_fitness),
                'mean_fitness': float(mean_fitness),
                'std_fitness': float(std_fitness),
                'best_task_fitness': float(best_fitness_task),
                'best_overall_fitness': float(best_fitness_overall),
                'elapsed_time': elapsed,
            })
            
            if gen % log_interval == 0 or gen == gens_per_task - 1:
                elapsed = time.time() - start_time
                
                print(f"Task {task_idx+1} Gen {gen:4d} | "
                      f"Best: {gen_best_fitness:8.2f} | "
                      f"Mean: {mean_fitness:8.2f} | "
                      f"{task_mod}: {current_multiplier:6.2f} | "
                      f"Time: {elapsed:6.1f}s")
            
            total_gen += 1
        
        print(f"\nTask {task_idx + 1} complete! Best fitness: {best_fitness_task:.2f}")
        
        # Save checkpoint at end of task
        if task_best_params is not None:
            # Create task label from multiplier
            task_label = f"{task_mod}_{current_multiplier:.2f}".replace(".", "p")
            checkpoint_path = os.path.join(checkpoint_dir, f"task_{task_idx:02d}_{task_label}.pkl")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'flat_params': task_best_params,
                    'fitness': float(best_fitness_task),
                    'task_idx': task_idx,
                    'task_mod': task_mod,
                    'multiplier': current_multiplier,
                    'generation': total_gen,
                    'config': {
                        'env_name': env_name,
                        'hidden_dims': hidden_dims,
                        'obs_dim': obs_dim,
                        'action_dim': action_dim,
                    }
                }, f)
            print(f"  Checkpoint saved: {checkpoint_path}")
            
            # Save GIFs at end of each task
            try:
                task_gifs_dir = os.path.join(output_dir, "gifs", f"task_{task_idx:02d}_{task_label}")
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
                        state = jit_step(state, action)
                        trajectory.append(state)
                        total_reward += float(state.reward)
                        if state.done:
                            break
                    
                    # Render every 2nd frame
                    images = env.render(trajectory[::2], height=240, width=320, camera="side")
                    gif_path = os.path.join(task_gifs_dir, f"trial{gif_idx}_reward{total_reward:.0f}.gif")
                    imageio.mimsave(gif_path, images, fps=30, loop=0)
                
                print(f"  Saved {num_gifs} GIFs to: {task_gifs_dir}")
            except Exception as e:
                print(f"  Warning: Failed to save GIFs for task {task_idx}: {e}")
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "=" * 60)
    print("Continual Learning Complete!")
    print("=" * 60)
    print(f"  Environment: {env_name}")
    print(f"  Total tasks: {num_tasks}")
    print(f"  Total generations: {total_gen}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Time per generation: {total_time/total_gen:.2f}s")
    print(f"  Best fitness achieved: {best_fitness_overall:.2f}")
    
    # Save best parameters
    if best_params_overall is not None:
        best_params_unflat = unflatten_params(best_params_overall, param_template)
        save_path = os.path.join(output_dir, f"best_{env_name.lower()}_continual_{mod_suffix}_policy.pkl")
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'params': best_params_unflat,
                'flat_params': best_params_overall,
                'fitness': float(best_fitness_overall),
                'config': {
                    'env_name': env_name,
                    'hidden_dims': hidden_dims,
                    'obs_dim': obs_dim,
                    'action_dim': action_dim,
                    'num_tasks': num_tasks,
                    'task_mod': task_mod,
                    'task_multipliers': list(multiplier_list) if not isinstance(multiplier_list, list) else multiplier_list,
                    'continual': True,
                }
            }, f)
        print(f"  Best policy saved to: {save_path}")
    
    # Run final evaluation with 100 trials, GIFs, and histogram
    # Use the last multiplier value for final eval
    if best_params_overall is not None:
        key, eval_key = jax.random.split(key)
        final_multiplier = float(multiplier_list[-1])
        eval_save_dir = os.path.join(output_dir, "eval_results")
        final_evaluation(
            env, policy, best_params_overall, param_template, eval_key, env_name, final_multiplier,
            num_eval_trials=100, max_steps=max_episode_steps, save_dir=eval_save_dir
        )
    
    # Save training metrics to CSV
    if training_metrics_list:
        import csv
        csv_path = os.path.join(output_dir, "training_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['generation', 'task', 'task_generation', 'multiplier', 'best_fitness', 'mean_fitness', 'std_fitness', 'best_task_fitness', 'best_overall_fitness', 'elapsed_time'])
            writer.writeheader()
            writer.writerows(training_metrics_list)
        print(f"  Training metrics saved to: {csv_path}")
    
    # Finish wandb run
    wandb.finish()
    
    return best_fitness_overall, best_params_overall


if __name__ == "__main__":
    main()
