"""
Train GA on Gymnax environments (CONTINUAL).

Task changes every 200 generations by either:
  - Adding observation noise (task_type=noise)
  - Varying an environment parameter (task_type=param):
      CartPole: gravity [0.98, 98.0] (default 9.8, factor of 10)
      MountainCar: gravity [0.000833, 0.0075] (default 0.0025, factor of 3)
      Acrobot: link_length_1 [0.5, 2.0] (default 1.0)

Supports CartPole-v1, Acrobot-v1, MountainCar-v0.

Usage:
    python train_GA_gymnax_continual.py --env CartPole-v1 --gpus 0
    python train_GA_gymnax_continual.py --env CartPole-v1 --task_type param --gpus 0
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
# SimpleGA imported dynamically based on --ga_version flag
import gymnax
import time
import pickle
import wandb
import numpy as np
import imageio
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Solved thresholds for Speed-Up (SU) metric
SOLVED_THRESHOLDS = {
    'CartPole-v1': 475,
    'Acrobot-v1': -70,
    'MountainCar-v0': -110,
}


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
        "num_generations": 2000,  # 10 tasks x 200 gens
        "pop_size": 512,
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 10,
        "task_interval": 200,
        "num_tasks": 10,
    },
    "Acrobot-v1": {
        "num_generations": 2000,
        "pop_size": 512,
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 10,
        "task_interval": 200,
        "num_tasks": 10,
    },
    "MountainCar-v0": {
        "num_generations": 2000,
        "pop_size": 512,
        "hidden_dims": (16, 16),
        "episode_length": 500,
        "num_evals": 10,
        "task_interval": 200,
        "num_tasks": 10,
    },
}


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

def make_scoring_fn(env, policy, param_template, episode_length, num_evals=10):
    """Create scoring function for continual learning.
    
    Accepts env_params and noise_vector as arguments (not closure-captured)
    so they can change across tasks.
    
    Evaluates each individual with num_evals trials:
    - Returns fitness from FIRST trial only (for selection)
    - Returns mean fitness across all trials (for logging/tracking)
    """
    
    def evaluate_single(flat_params, eval_key, noise_vector, env_params):
        params = unflatten_params(flat_params, param_template)
        obs, state = env.reset(eval_key, env_params)
        
        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
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
    
    vmapped_eval = jax.vmap(evaluate_single, in_axes=(0, 0, None, None))
    
    @jax.jit
    def scoring_fn(flat_genotypes, key, noise_vector, env_params):
        """Returns (fitness_for_selection, mean_fitness_for_logging)."""
        pop_size = flat_genotypes.shape[0]
        
        # Always evaluate with num_evals trials per individual
        all_keys = random.split(key, pop_size * num_evals)
        flat_params_repeated = jnp.repeat(flat_genotypes, num_evals, axis=0)
        all_fitnesses = vmapped_eval(flat_params_repeated, all_keys, noise_vector, env_params)
        all_fitnesses = all_fitnesses.reshape(pop_size, num_evals)
        
        # Fitness for selection: use FIRST trial only
        fitnesses = all_fitnesses[:, 0]
        
        # Mean fitness for logging/tracking
        mean_fitnesses = jnp.mean(all_fitnesses, axis=1)
        
        return fitnesses, mean_fitnesses
    
    return scoring_fn


def rollout_for_gif(env, env_params, policy, flat_params, param_template, episode_length, key, noise_vector=None):
    """Rollout for GIF generation."""
    if noise_vector is None:
        noise_vector = jnp.zeros(env.observation_space(env_params).shape)
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
    parser = argparse.ArgumentParser(description='GA on Gymnax (Continual)')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'])
    parser.add_argument('--num_generations', type=int, default=None)
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--elite_ratio', type=float, default=0.5)
    parser.add_argument('--mutation_std', type=float, default=0.5)
    parser.add_argument('--num_evals', type=int, default=None)
    parser.add_argument('--task_interval', type=int, default=200)
    parser.add_argument('--noise_range', type=float, default=1.0,
                        help='Scale for observation noise (task_type=noise)')
    parser.add_argument('--noise_type', type=str, default='normal',
                        choices=['normal', 'uniform'],
                        help='Noise distribution: normal (Gaussian) or uniform')
    parser.add_argument('--task_type', type=str, default='noise',
                        choices=['noise', 'param'],
                        help='How tasks differ: observation noise or env parameter variation')
    parser.add_argument('--param_range', type=float, nargs=2, default=None,
                        help='Range [min, max] for param sampling (task_type=param). Default: env-specific.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='continual_neuroevolution_gymnax')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--ga_version', type=str, default='evosax',
                        choices=['kinetix', 'evosax'],
                        help='GA implementation: kinetix (local) or evosax (pip)')
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    
    env_name = args.env
    seed = args.seed + args.trial  # Different seed per trial
    trial = args.trial
    
    # Get env-specific config
    cfg = ENV_CONFIGS[env_name]
    num_generations = args.num_generations or cfg["num_generations"]
    pop_size = args.pop_size or cfg["pop_size"]
    hidden_dims = cfg["hidden_dims"]
    episode_length = cfg["episode_length"]
    num_evals = args.num_evals or cfg["num_evals"]
    task_interval = args.task_interval
    noise_range = args.noise_range
    noise_type = args.noise_type
    task_type = args.task_type
    
    # Environment-specific parameter variation config
    PARAM_CONFIGS = {
        'CartPole-v1': {'param_name': 'gravity', 'default_val': 9.8, 'default_range': [0.98, 98.0]},
        'MountainCar-v0': {'param_name': 'gravity', 'default_val': 0.0025, 'default_range': [0.000833, 0.0075]},
        'Acrobot-v1': {'param_name': 'link_length_1', 'default_val': 1.0, 'default_range': [0.5, 2.0]},
    }
    param_cfg = PARAM_CONFIGS[env_name]
    param_name = param_cfg['param_name']
    param_range = args.param_range if args.param_range is not None else param_cfg['default_range']
    
    output_dir = args.output_dir or f"projects/gymnax/ga_{env_name}_continual_{task_type}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)
    gifs_dir = os.path.join(output_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, "train.log")
    sys.stdout = Tee(log_file)
    
    print("=" * 60)
    print(f"GA on {env_name} (CONTINUAL)")
    print("=" * 60)
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}")
    print(f"  Task interval: {task_interval} gens")
    print(f"  Task type: {task_type}")
    if task_type == 'noise':
        print(f"  Noise range: {noise_range}")
    else:
        print(f"  Param: {param_name}, range: {param_range}")
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
    
    # Pre-generate deterministic task sequence (same across methods for same trial)
    num_tasks = num_generations // task_interval
    task_rng = jax.random.key(trial * 7919)  # Separate RNG, deterministic per trial
    if task_type == 'noise':
        task_noise_vectors = [jnp.zeros((obs_dim,))]  # Task 0 = no noise (identical to non-continual)
        for t in range(1, num_tasks):
            task_rng, noise_key = random.split(task_rng)
            if noise_type == 'normal':
                nv = random.normal(noise_key, (obs_dim,)) * noise_range
            else:
                nv = random.uniform(noise_key, (obs_dim,), minval=-noise_range, maxval=noise_range)
            task_noise_vectors.append(nv)
        print(f"  Pre-generated {num_tasks} noise vectors (task 0 = zero noise)")
    elif task_type == 'param':
        task_param_values = [param_cfg['default_val']]  # Task 0 = default
        for t in range(1, num_tasks):
            task_rng, param_key = random.split(task_rng)
            pv = float(random.uniform(param_key, minval=param_range[0], maxval=param_range[1]))
            task_param_values.append(pv)
        print(f"  Pre-generated {num_tasks} param values: {task_param_values}")
    
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create policy
    key, init_key = jax.random.split(key)
    policy, param_template = create_policy_network(init_key, obs_dim, action_dim, hidden_dims)
    flat_params = get_flat_params(param_template)
    num_params = flat_params.shape[0]
    print(f"  Network: {hidden_dims}, {num_params} params")
    
    # Create scoring function (env_params passed as argument, not captured in closure)
    scoring_fn = make_scoring_fn(env, policy, param_template, episode_length, num_evals)
    
    # Initialize wandb
    config = {
        'env': env_name, 'num_generations': num_generations,
        'pop_size': pop_size, 'seed': seed, 'trial': trial,
        'elite_ratio': args.elite_ratio, 'mutation_std': args.mutation_std,
        'num_evals': num_evals, 'hidden_dims': hidden_dims,
        'task_interval': task_interval, 'noise_range': noise_range,
        'noise_type': noise_type,
        'task_type': task_type, 'param_name': param_name, 'param_range': param_range,
        'continual': True,
    }
    wandb.init(project=args.wandb_project, config=config,
               name=f"ga_{env_name}_continual_{task_type}_pop{pop_size}_trial{trial}", reinit=True)
    
    # Setup GA
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(np.array(devices), axis_names=('p',))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())
    parallel_sharding = NamedSharding(mesh, PartitionSpec('p'))
    
    if pop_size % num_devices != 0:
        old_pop_size = pop_size
        pop_size = (pop_size // num_devices + 1) * num_devices
        print(f"  Warning: Adjusted pop_size from {old_pop_size} to {pop_size}")
    
    # Select GA implementation based on flag
    ga_version = args.ga_version
    print(f"  Using GA version: {ga_version}")
    
    if ga_version == 'kinetix':
        from simple_ga import SimpleGA
        ga = SimpleGA(
            popsize=pop_size,
            num_dims=num_params,
            elite_ratio=args.elite_ratio,
            sigma_init=args.mutation_std,
            init_min=-0.1,
            init_max=0.1,
        )
        ga_params = ga.default_params
        
        key, state_init_key = random.split(key)
        ga_state = ga.initialize(state_init_key, ga_params)
        
        @jax.jit
        def jit_ask(key, state, params):
            population, new_state = ga.ask(key, state, params)
            population = jax.device_put(population, parallel_sharding)
            return population, new_state
        
        @jax.jit
        def jit_tell(population, fitness, state, params):
            return ga.tell(population, fitness, state, params)
    else:  # evosax
        from evosax.algorithms import SimpleGA
        std_schedule = lambda gen: args.mutation_std
        ga = SimpleGA(
            population_size=pop_size,
            solution=jnp.zeros(num_params),
            std_schedule=std_schedule,
        )
        ga.elite_ratio = args.elite_ratio
        ga_params = jax.device_put(ga.default_params, replicate_sharding)
        
        key, init_key = random.split(key)
        init_population = random.normal(init_key, (pop_size, num_params)) * 0.1
        init_fitness = jnp.full(pop_size, jnp.inf)
        
        key, state_init_key = random.split(key)
        ga_state = jax.jit(ga.init, out_shardings=replicate_sharding)(
            state_init_key, init_population, init_fitness, ga_params
        )
        
        @jax.jit
        def jit_ask(key, state, params):
            population, new_state = ga.ask(key, state, params)
            population = jax.device_put(population, parallel_sharding)
            return population, new_state
        
        @jax.jit
        def jit_tell_evosax(key, population, fitness, state, params):
            return ga.tell(key, population, fitness, state, params)
    
    # Initialize task 0
    current_task = 0
    noise_vector = jnp.zeros((obs_dim,))
    
    if task_type == 'noise':
        noise_vector = task_noise_vectors[0]
        print(f"\n  Task 0 noise vector ({noise_type}): {jax.device_get(noise_vector)}")
        print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")
    elif task_type == 'param':
        param_val = task_param_values[0]
        env_params = env_params.replace(**{param_name: param_val})
        print(f"\n  Task 0 {param_name}: {param_val:.4f}")
    
    # Warmup JIT
    print("\nJIT compiling...")
    key, warmup_key, warmup_ask_key = random.split(key, 3)
    warmup_pop, _ = jit_ask(warmup_ask_key, ga_state, ga_params)
    _ = scoring_fn(warmup_pop, warmup_key, noise_vector, env_params)
    print("  JIT compilation complete!")
    
    # Training loop
    best_overall_fitness = -float('inf')
    best_params = None
    # Track per-task best separately (for GIF evaluation)
    task_best_fitness = -float('inf')
    task_best_params = None
    start_time = time.time()
    render_fn = get_render_fn(env_name)
    
    # Metrics tracking
    solved_threshold = SOLVED_THRESHOLDS.get(env_name)
    task_start_gen = 0
    task_gens_to_threshold = None  # gen (relative to task start) when threshold first reached
    all_metrics = []  # list of per-task metric dicts
    
    # Zero-shot eval for task 0: evaluate random init on task 0
    # Use the initial population's best individual
    key, zt_init_key = random.split(key)
    warmup_pop_zt, _ = jit_ask(zt_init_key, ga_state, ga_params)
    warmup_pop_host = jax.device_get(warmup_pop_zt)
    # Just use first individual (all random, no "best" yet)
    zt_params = warmup_pop_host[0]
    zt_rewards = []
    for eval_trial in range(10):
        key, eval_key_trial = random.split(key)
        _, trial_reward = rollout_for_gif(
            env, env_params, policy, zt_params, param_template,
            episode_length, eval_key_trial, noise_vector
        )
        zt_rewards.append(trial_reward)
    task_zt_mean = float(np.mean(zt_rewards))
    task_zt_std = float(np.std(zt_rewards))
    print(f"  Task 0 zero-shot: {task_zt_mean:.2f} +/- {task_zt_std:.2f}")
    
    print(f"\nStarting continual training...")
    
    for gen in range(num_generations):
        # Check if we need to switch task
        if gen > 0 and gen % task_interval == 0:
            # Use best from the last generation of this task (not best across all task generations)
            eval_params = population_host[int(np.argmax(mean_fitness_host))].copy()
            
            # Per-task evaluation with 10 trials
            print(f"\n  Task {current_task} final evaluation (10 trials)...")
            task_eval_rewards = []
            for eval_trial in range(10):
                key, eval_key_trial = random.split(key)
                _, trial_reward = rollout_for_gif(
                    env, env_params, policy, eval_params, param_template,
                    episode_length, eval_key_trial, noise_vector
                )
                task_eval_rewards.append(trial_reward)
            task_eval_mean = float(np.mean(task_eval_rewards))
            task_eval_std = float(np.std(task_eval_rewards))
            print(f"  Task {current_task} eval: {task_eval_mean:.2f} +/- {task_eval_std:.2f}")
            wandb.summary[f"task_{current_task}_eval_mean"] = task_eval_mean
            wandb.summary[f"task_{current_task}_eval_std"] = task_eval_std
            
            # Create task subfolder and save 10 GIFs
            task_gif_dir = os.path.join(gifs_dir, f"task{current_task}")
            os.makedirs(task_gif_dir, exist_ok=True)
            
            try:
                for gif_idx in range(10):
                    key, gif_key = random.split(key)
                    obs_list, total_reward = rollout_for_gif(
                        env, env_params, policy, eval_params, param_template, 
                        episode_length, gif_key, noise_vector
                    )
                    
                    frames = []
                    fig, ax = None, None
                    for idx, obs in enumerate(obs_list[::2]):
                        step = idx * 2  # Actual step in episode
                        frame, fig, ax = render_fn(obs, fig, ax, step=step)
                        frames.append(frame)
                    plt.close(fig)
                    
                    gif_path = os.path.join(task_gif_dir, f"task{current_task}_rollout_{gif_idx:02d}_reward{total_reward:.0f}.gif")
                    imageio.mimsave(gif_path, frames, fps=30, loop=0)
                
                print(f"  Saved 10 GIFs for task {current_task} in {task_gif_dir}")
            except Exception as e:
                print(f"  Warning: Failed to save GIFs: {e}")
            
            # Save per-task checkpoint
            task_ckpt_path = os.path.join(checkpoints_dir, f"task_{current_task}.pkl")
            ckpt_data = {
                'flat_params': np.array(eval_params),
                'task_idx': current_task,
                'task_type': task_type,
                'generation': gen,
                'best_fitness': float(task_best_fitness),
                'eval_mean': task_eval_mean,
                'eval_std': task_eval_std,
                'zero_shot_eval_mean': task_zt_mean,
                'zero_shot_eval_std': task_zt_std,
                'gens_to_threshold': task_gens_to_threshold,
            }
            if task_type == 'noise':
                ckpt_data['noise_vector'] = jax.device_get(noise_vector)
            elif task_type == 'param':
                ckpt_data['param_name'] = param_name
                ckpt_data['param_value'] = float(getattr(env_params, param_name))
            with open(task_ckpt_path, 'wb') as f:
                pickle.dump(ckpt_data, f)
            print(f"    Saved task checkpoint: {task_ckpt_path}")
            
            # Store per-task metrics
            all_metrics.append({
                'task_idx': current_task,
                'eval_mean': task_eval_mean,
                'eval_std': task_eval_std,
                'zero_shot_eval_mean': task_zt_mean,
                'zero_shot_eval_std': task_zt_std,
                'gens_to_threshold': task_gens_to_threshold,
            })

            # Switch to new task
            current_task += 1
            task_start_gen = gen
            task_gens_to_threshold = None
            if task_type == 'noise':
                noise_vector = task_noise_vectors[current_task]
                print(f"\n>>> Task {current_task} started at gen {gen}")
                print(f"  Full noise ({noise_type}): {jax.device_get(noise_vector)}")
                print(f"  Noise magnitude: {float(jnp.linalg.norm(noise_vector)):.4f}")
            elif task_type == 'param':
                param_val = task_param_values[current_task]
                env_params = env_params.replace(**{param_name: param_val})
                print(f"\n>>> Task {current_task} started at gen {gen}")
                print(f"  {param_name}: {param_val:.4f}")
            
            # Zero-shot evaluation on new task (before training)
            zt_rewards = []
            for eval_trial in range(10):
                key, eval_key_trial = random.split(key)
                _, trial_reward = rollout_for_gif(
                    env, env_params, policy, eval_params, param_template,
                    episode_length, eval_key_trial, noise_vector
                )
                zt_rewards.append(trial_reward)
            task_zt_mean = float(np.mean(zt_rewards))
            task_zt_std = float(np.std(zt_rewards))
            print(f"  Task {current_task} zero-shot: {task_zt_mean:.2f} +/- {task_zt_std:.2f}")
            wandb.summary[f"task_{current_task}_zero_shot_mean"] = task_zt_mean
            wandb.summary[f"task_{current_task}_zero_shot_std"] = task_zt_std
            
            # Reset task-specific best tracking for new task
            task_best_fitness = -float('inf')
            task_best_params = None
        
        key, ask_key, eval_key, tell_key = random.split(key, 4)
        
        population, ga_state = jit_ask(ask_key, ga_state, ga_params)
        fitness, mean_fitness = scoring_fn(population, eval_key, noise_vector, env_params)
        
        mean_fitness_host = jax.device_get(mean_fitness)
        population_host = jax.device_get(population)
        
        # evosax SimpleGA MINIMIZES by default, so negate fitness (use single-trial for selection)
        if ga_version == 'kinetix':
            ga_state = jit_tell(population, -fitness, ga_state, ga_params)
        else:
            ga_state, _ = jit_tell_evosax(tell_key, population, -fitness, ga_state, ga_params)
        
        # Track using mean-of-10 fitness
        gen_best = float(np.max(mean_fitness_host))
        gen_mean = float(np.mean(mean_fitness_host))
        best_idx = int(np.argmax(mean_fitness_host))
        
        # Update task-specific best (for GIF evaluation)
        if gen_best > task_best_fitness:
            task_best_fitness = gen_best
            task_best_params = population_host[best_idx].copy()
        
        # Track generations to threshold (SU metric)
        if solved_threshold is not None and task_gens_to_threshold is None:
            if gen_best >= solved_threshold:
                task_gens_to_threshold = gen - task_start_gen
                print(f"  Task {current_task} reached threshold {solved_threshold} at gen {gen} (gens_in_task={task_gens_to_threshold})")
        
        # Update overall best (for checkpoint)
        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best
            best_params = population_host[best_idx].copy()
        
        log_dict = {
            "generation": gen, "best_fitness": gen_best,
            "mean_fitness": gen_mean, "best_overall": best_overall_fitness,
            "task": current_task,
        }
        if task_type == 'noise':
            log_dict["noise_magnitude"] = float(jnp.linalg.norm(noise_vector))
        elif task_type == 'param':
            log_dict[param_name] = float(getattr(env_params, param_name))
        wandb.log(log_dict)
        
        if gen % args.log_interval == 0 or gen == num_generations - 1:
            print(f"Gen {gen:4d} | Task {current_task} | Best: {gen_best:8.2f} | Mean: {gen_mean:8.2f} | Overall: {best_overall_fitness:8.2f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Best overall: {best_overall_fitness:.2f}")
    print(f"  Total tasks: {current_task + 1}")
    
    # Use best from final generation of this task (not best across all task generations)
    final_best_params = population_host[int(np.argmax(mean_fitness_host))].copy()
    
    # Per-task evaluation with 10 trials for final task
    print(f"\n  Task {current_task} final evaluation (10 trials)...")
    task_eval_rewards = []
    for eval_trial in range(10):
        key, eval_key_trial = random.split(key)
        _, trial_reward = rollout_for_gif(
            env, env_params, policy, final_best_params, param_template,
            episode_length, eval_key_trial, noise_vector
        )
        task_eval_rewards.append(trial_reward)
    task_eval_mean = float(np.mean(task_eval_rewards))
    task_eval_std = float(np.std(task_eval_rewards))
    print(f"  Task {current_task} eval: {task_eval_mean:.2f} +/- {task_eval_std:.2f}")
    wandb.summary[f"task_{current_task}_eval_mean"] = task_eval_mean
    wandb.summary[f"task_{current_task}_eval_std"] = task_eval_std
    
    # Save 10 GIFs for final task
    task_gif_dir = os.path.join(gifs_dir, f"task{current_task}")
    os.makedirs(task_gif_dir, exist_ok=True)
    
    try:
        for gif_idx in range(10):
            key, gif_key = random.split(key)
            obs_list, total_reward = rollout_for_gif(
                env, env_params, policy, final_best_params, param_template, 
                episode_length, gif_key, noise_vector
            )
            
            frames = []
            fig, ax = None, None
            for idx, obs in enumerate(obs_list[::2]):
                step = idx * 2  # Actual step in episode
                frame, fig, ax = render_fn(obs, fig, ax, step=step)
                frames.append(frame)
            plt.close(fig)
            
            gif_path = os.path.join(task_gif_dir, f"task{current_task}_rollout_{gif_idx:02d}_reward{total_reward:.0f}.gif")
            imageio.mimsave(gif_path, frames, fps=30, loop=0)
        
        print(f"Saved 10 GIFs for final task {current_task} in {task_gif_dir}")
    except Exception as e:
        print(f"Warning: Failed to save final GIFs: {e}")
    
    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"ga_{env_name}_continual_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': np.array(best_params) if best_params is not None else None,
            'param_template': param_template,
            'best_fitness': best_overall_fitness,
            'config': config,
            'final_task': current_task,
        }, f)
    print(f"Saved: {ckpt_path}")
    
    # Save final task checkpoint for KL divergence analysis
    task_ckpt_path = os.path.join(checkpoints_dir, f"task_{current_task}.pkl")
    final_ckpt_data = {
        'flat_params': np.array(final_best_params),
        'task_idx': current_task,
        'task_type': task_type,
        'generation': num_generations,
        'best_fitness': float(task_best_fitness),
        'eval_mean': task_eval_mean,
        'eval_std': task_eval_std,
        'zero_shot_eval_mean': task_zt_mean,
        'zero_shot_eval_std': task_zt_std,
        'gens_to_threshold': task_gens_to_threshold,
    }
    if task_type == 'noise':
        final_ckpt_data['noise_vector'] = jax.device_get(noise_vector)
    elif task_type == 'param':
        final_ckpt_data['param_name'] = param_name
        final_ckpt_data['param_value'] = float(getattr(env_params, param_name))
    with open(task_ckpt_path, 'wb') as f:
        pickle.dump(final_ckpt_data, f)
    print(f"Saved final task checkpoint: {task_ckpt_path}")
    
    # Store final task metrics
    all_metrics.append({
        'task_idx': current_task,
        'eval_mean': task_eval_mean,
        'eval_std': task_eval_std,
        'zero_shot_eval_mean': task_zt_mean,
        'zero_shot_eval_std': task_zt_std,
        'gens_to_threshold': task_gens_to_threshold,
    })
    
    # Compute aggregate metrics and save to YAML
    num_tasks = len(all_metrics)
    num_solved = sum(1 for m in all_metrics if solved_threshold is not None and m['eval_mean'] >= solved_threshold)
    success_rate = num_solved / num_tasks if num_tasks > 0 else 0.0
    zt_values = [m['zero_shot_eval_mean'] for m in all_metrics]
    
    summary_metrics = {
        'method': 'ga',
        'env': env_name,
        'task_type': task_type,
        'num_tasks': num_tasks,
        'solved_threshold': solved_threshold,
        'success_rate': success_rate,
        'num_solved': num_solved,
        'zero_shot_transfer_mean': float(np.mean(zt_values)),
        'zero_shot_transfer_std': float(np.std(zt_values)),
        'per_task': all_metrics,
    }
    metrics_path = os.path.join(output_dir, "metrics.yaml")
    with open(metrics_path, 'w') as f:
        yaml.dump(summary_metrics, f, default_flow_style=False)
    print(f"Saved metrics: {metrics_path}")

    wandb.finish()
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
