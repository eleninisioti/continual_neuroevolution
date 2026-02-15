"""
Train OpenES on Kinetix environments (non-continual).

Uses pixel-based observations by default, multi-discrete action space.
Runs on 20 medium h-tasks (h0-h19).

Usage:
    python train_ES_kinetix.py --env h0_unicycle --gpus 0
    python train_ES_kinetix.py --env h5_angry_birds --gpus 1 --symbolic
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
import optax
from evosax.algorithms import Open_ES
import time
import pickle
import json
import wandb
import numpy as np
import imageio

# Kinetix imports
from kinetix.environment.env import make_kinetix_env
from kinetix.environment.ued.ued import make_reset_fn_from_config
from kinetix.environment.utils import ActionType, ObservationType
from kinetix.environment.env_state import EnvParams, StaticEnvParams
from kinetix.util.config import normalise_config
from kinetix.util.saving import load_from_json_file
from kinetix.render.renderer_pixels import make_render_pixels
from flax.serialization import to_state_dict
import yaml


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
# Policy Network for Kinetix (Multi-Discrete Actions)
# ============================================================================

class ConvPolicy(nn.Module):
    """CNN policy for pixel observations with multi-discrete actions."""
    action_dims: tuple  # e.g., (10, 2) for 10 actions with 2 options each
    
    @nn.compact
    def __call__(self, obs):
        """
        obs: PixelObservation with .image (H, W, C) and .global_info (D,)
        """
        # Handle PixelObservation: extract image
        if hasattr(obs, 'image'):
            x = obs.image  # (H, W, C)
            global_info = obs.global_info
        else:
            x = obs
            global_info = None
        
        # CNN for image
        x = nn.Conv(features=16, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape(-1)  # Flatten
        
        # Concatenate global info if available
        if global_info is not None:
            x = jnp.concatenate([x, global_info])
        
        # MLP head
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        
        # Output heads for each action dimension
        outputs = []
        for dim in self.action_dims:
            out = nn.Dense(dim)(x)
            outputs.append(out)
        
        return outputs  # List of logits for each action dim


class MLPPolicy(nn.Module):
    """MLP policy for symbolic observations with multi-discrete actions."""
    hidden_dims: tuple = (64, 64)
    action_dims: tuple = (10, 2)  # Multi-discrete action space
    
    @nn.compact
    def __call__(self, x):
        # Flatten if needed
        if x.ndim > 1:
            x = x.reshape(-1)
        
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)
        
        # Output heads for each action dimension
        outputs = []
        for dim in self.action_dims:
            out = nn.Dense(dim)(x)
            outputs.append(out)
        
        return outputs  # List of logits for each action dim


def get_flat_params(params):
    flat_params, _ = flatten_util.ravel_pytree(params)
    return flat_params


def unflatten_params(flat_params, param_template):
    _, unravel_fn = flatten_util.ravel_pytree(param_template)
    return unravel_fn(flat_params)


def create_policy_network(key, dummy_obs, action_dims, use_cnn=True, hidden_dims=(64, 64)):
    """Create policy network and return (network, params)."""
    if use_cnn:
        policy = ConvPolicy(action_dims=action_dims)
    else:
        policy = MLPPolicy(hidden_dims=hidden_dims, action_dims=action_dims)
    
    params = policy.init(key, dummy_obs)
    return policy, params


# ============================================================================
# Environment Setup
# ============================================================================

# Default config for kinetix
KINETIX_CONFIG = {
    "num_generations": 4000,  # ~200*20 like inspiration hyperparams
    "pop_size": 1024,
    "hidden_dims": (64, 64),
    "episode_length": 1000,
    "num_evals": 1,
    "sigma": 0.01,
    "learning_rate": 0.01,
}

# 20 medium h-tasks
ENVIRONMENTS = [
    "h0_unicycle",
    "h1_car_left",
    "h2_car_right",
    "h3_car_thrust",
    "h4_thrust_the_needle",
    "h5_angry_birds",
    "h6_thrust_over",
    "h7_car_flip",
    "h8_weird_vehicle",
    "h9_spin_the_right_way",
    "h10_thrust_right_easy",
    "h11_thrust_left_easy",
    "h12_thrustfall_left",
    "h13_thrustfall_right",
    "h14_thrustblock",
    "h15_thrustshoot",
    "h16_thrustcontrol_right",
    "h17_thrustcontrol_left",
    "h18_thrust_right_very_easy",
    "h19_thrust_left_very_easy",
]


def load_kinetix_env(env_name, use_pixels=True):
    """Load a kinetix environment from JSON file."""
    # Load level from JSON
    level_path = f"m/{env_name}.json"
    env_state, static_env_params, env_params = load_from_json_file(level_path)
    
    # Create config
    config = {
        "observation_type": ObservationType.PIXELS if use_pixels else ObservationType.SYMBOLIC_FLAT,
        "action_type": ActionType.MULTI_DISCRETE,
        "train_level_mode": "list",
        "train_levels_list": [level_path],
    }
    
    # Create reset function
    reset_fn = make_reset_fn_from_config(config, env_params, static_env_params)
    
    # Create environment
    env = make_kinetix_env(
        observation_type=config["observation_type"],
        action_type=config["action_type"],
        reset_fn=reset_fn,
        env_params=env_params,
        static_env_params=static_env_params,
    )
    
    return env, env_params, static_env_params, env_state


def get_action_dims(env):
    """Get action dimensions for multi-discrete action space."""
    # Kinetix multi-discrete: typically 10 action components with varying options
    # The action space is defined by the environment's action_space
    action_space = env.action_space()
    if hasattr(action_space, 'nvec'):
        return tuple(action_space.nvec)
    else:
        # Fallback: assume 10 actions with 2 options each
        return (2,) * 10


# ============================================================================
# Scoring Function
# ============================================================================

def make_scoring_fn(env, env_params, env_state_init, policy, param_template, episode_length, num_evals=1):
    """Create JIT-compiled scoring function for kinetix."""
    
    def evaluate_single(flat_params, eval_key):
        params = unflatten_params(flat_params, param_template)
        reset_key, rollout_key = random.split(eval_key)
        
        # Reset environment
        obs, state = env.reset(reset_key, env_params, override_reset_state=env_state_init)
        
        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
            
            # Get action logits
            action_logits = policy.apply(params, obs)
            
            # Multi-discrete action: argmax on each dimension
            actions = jnp.array([jnp.argmax(logits) for logits in action_logits])
            
            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(step_key, state, actions, env_params)
            
            # Accumulate reward only if not done
            total_reward = total_reward + reward * (1.0 - done_flag)
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
            
            return (next_obs, next_state, total_reward, done_flag, key), None
        
        (_, _, total_reward, _, _), _ = jax.lax.scan(
            step_fn, (obs, state, 0.0, 0.0, rollout_key), None, length=episode_length
        )
        return total_reward
    
    vmapped_eval = jax.vmap(evaluate_single)
    
    @jax.jit
    def scoring_fn(flat_genotypes, key):
        """Returns (fitness_for_selection, mean_fitness_for_logging)."""
        pop_size = flat_genotypes.shape[0]
        
        # Evaluate with num_evals trials per individual
        all_keys = random.split(key, pop_size * num_evals)
        flat_params_repeated = jnp.repeat(flat_genotypes, num_evals, axis=0)
        all_fitnesses = vmapped_eval(flat_params_repeated, all_keys)
        all_fitnesses = all_fitnesses.reshape(pop_size, num_evals)
        
        # Fitness for selection: use FIRST trial only
        fitnesses = all_fitnesses[:, 0]
        
        # Mean fitness for logging
        mean_fitnesses = jnp.mean(all_fitnesses, axis=1)
        
        return fitnesses, mean_fitnesses
    
    return scoring_fn


def rollout_for_gif(env, env_params, env_state_init, static_env_params, policy, flat_params, param_template, episode_length, key):
    """Rollout for evaluation/GIF generation."""
    params = unflatten_params(flat_params, param_template)
    
    reset_key, rollout_key = random.split(key)
    obs, state = env.reset(reset_key, env_params, override_reset_state=env_state_init)
    
    # Create renderer
    render_static_params = static_env_params.replace(downscale=4)
    pixel_renderer = jax.jit(make_render_pixels(env_params, render_static_params))
    
    frames = []
    total_reward = 0.0
    
    for step in range(episode_length):
        # Render frame (transpose and flip to correct orientation)
        frame = np.array(pixel_renderer(state))
        frame = frame.transpose(1, 0, 2)[::-1].astype(np.uint8)
        frames.append(frame)
        
        # Get action
        action_logits = policy.apply(params, obs)
        actions = jnp.array([int(jnp.argmax(logits)) for logits in action_logits])
        
        rollout_key, step_key = random.split(rollout_key)
        obs, state, reward, done, _ = env.step(step_key, state, actions, env_params)
        total_reward += float(reward)
        
        if bool(done):
            # Add final frame
            frame = np.array(pixel_renderer(state))
            frame = frame.transpose(1, 0, 2)[::-1].astype(np.uint8)
            frames.append(frame)
            break
    
    return frames, total_reward


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='OpenES on Kinetix (Non-Continual)')
    parser.add_argument('--env', type=str, default='h0_unicycle',
                        choices=ENVIRONMENTS)
    parser.add_argument('--symbolic', action='store_true',
                        help='Use symbolic observations instead of pixels')
    parser.add_argument('--num_generations', type=int, default=None)
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_evals', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--project_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='Kinetix-ES-noncontinual')
    parser.add_argument('--log_interval', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    
    env_name = args.env
    seed = args.seed
    trial = args.trial
    use_pixels = not args.symbolic
    obs_type = "pixels" if use_pixels else "symbolic"
    
    # Get config
    cfg = KINETIX_CONFIG
    num_generations = args.num_generations or cfg["num_generations"]
    pop_size = args.pop_size or cfg["pop_size"]
    hidden_dims = cfg["hidden_dims"]
    episode_length = cfg["episode_length"]
    num_evals = args.num_evals or cfg["num_evals"]
    sigma = args.sigma or cfg["sigma"]
    learning_rate = args.learning_rate or cfg["learning_rate"]
    
    # Output directory
    project_dir = args.project_dir or os.path.join(REPO_ROOT, "projects", "kinetix")
    output_dir = args.output_dir or os.path.join(project_dir, "vanilla", "es", env_name, f"trial_{trial}")
    os.makedirs(output_dir, exist_ok=True)
    gifs_dir = os.path.join(output_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, "train.log")
    tee_logger = Tee(log_file)
    sys.stdout = tee_logger
    
    print("=" * 60)
    print(f"OpenES on Kinetix: {env_name} (Non-Continual)")
    print("=" * 60)
    print(f"  Observation type: {obs_type}")
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}")
    print(f"  Sigma: {sigma}, LR: {learning_rate}")
    print(f"  Num evals: {num_evals}")
    print(f"  Output dir: {output_dir}")
    
    key = random.key(seed)
    
    # Load kinetix environment
    print(f"\nLoading kinetix environment: {env_name}...")
    env, env_params, static_env_params, env_state_init = load_kinetix_env(env_name, use_pixels=use_pixels)
    print(f"  Environment loaded successfully!")
    
    # Get observation and action dimensions
    key, reset_key = random.split(key)
    dummy_obs, _ = env.reset(reset_key, env_params, override_reset_state=env_state_init)
    action_dims = get_action_dims(env)
    
    print(f"  Action dims: {action_dims}")
    
    # Create policy
    key, init_key = random.split(key)
    policy, param_template = create_policy_network(
        init_key, dummy_obs, action_dims, 
        use_cnn=use_pixels, hidden_dims=hidden_dims
    )
    flat_params = get_flat_params(param_template)
    num_params = flat_params.shape[0]
    print(f"  Network: {'CNN' if use_pixels else 'MLP'}, {num_params} params")
    
    # Create scoring function
    print(f"\nCreating scoring function...")
    scoring_fn = make_scoring_fn(
        env, env_params, env_state_init, 
        policy, param_template, episode_length, num_evals
    )
    
    # Initialize wandb
    config = {
        'env': env_name, 'obs_type': obs_type,
        'num_generations': num_generations, 'pop_size': pop_size,
        'seed': seed, 'trial': trial,
        'sigma': sigma, 'learning_rate': learning_rate,
        'num_evals': num_evals, 'hidden_dims': hidden_dims,
        'num_params': num_params,
    }
    wandb.init(
        project=args.wandb_project, config=config,
        name=f"es_{env_name}_{obs_type}_trial{trial}", reinit=True
    )
    
    # Initialize ES with multi-GPU sharding
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(np.array(devices), axis_names=('p',))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())
    parallel_sharding = NamedSharding(mesh, PartitionSpec('p'))
    
    if pop_size % num_devices != 0:
        old_pop_size = pop_size
        pop_size = (pop_size // num_devices + 1) * num_devices
        print(f"  Warning: Adjusted pop_size from {old_pop_size} to {pop_size}")
    
    # Custom std schedule (constant)
    std_schedule = lambda gen: sigma
    
    # OpenES uses Adam optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    
    es = Open_ES(
        population_size=pop_size,
        solution=jnp.zeros(num_params),
        std_schedule=std_schedule,
        optimizer=optimizer,
        use_antithetic_sampling=True,
    )
    es_params = jax.device_put(es.default_params, replicate_sharding)
    
    key, init_key = random.split(key)
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
    
    # Warmup JIT
    print("\nJIT compiling...")
    key, warmup_key, warmup_ask_key = random.split(key, 3)
    warmup_pop, _ = jit_ask(warmup_ask_key, es_state, es_params)
    _, _ = scoring_fn(warmup_pop, warmup_key)
    print("  JIT compilation complete!")
    
    # Training loop
    best_overall_fitness = -float('inf')
    best_params = None
    start_time = time.time()
    
    print(f"\nStarting training...")
    
    for gen in range(num_generations):
        key, ask_key, eval_key, tell_key = random.split(key, 4)
        
        population, es_state = jit_ask(ask_key, es_state, es_params)
        fitness, mean_fitness = scoring_fn(population, eval_key)
        # evosax minimizes, so negate fitness for maximization
        es_state, _ = jit_tell(tell_key, population, -fitness, es_state, es_params)
        
        mean_fitness_host = jax.device_get(mean_fitness)
        
        # Track using mean fitness
        gen_best = float(np.max(mean_fitness_host))
        gen_mean = float(np.mean(mean_fitness_host))
        
        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best
            best_params = jax.device_get(es_state.mean)
        
        wandb.log({
            "generation": gen, "best_fitness": gen_best,
            "mean_fitness": gen_mean, "best_overall": best_overall_fitness,
        })
        
        if gen % args.log_interval == 0 or gen == num_generations - 1:
            elapsed = time.time() - start_time
            print(f"Gen {gen:4d} | Best: {gen_best:8.2f} | Mean: {gen_mean:8.2f} | "
                  f"Overall: {best_overall_fitness:8.2f} | Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Best overall: {best_overall_fitness:.2f}")
    
    # Use ES mean as the best solution if no improvement was found
    if best_params is None:
        best_params = jax.device_get(es_state.mean)
    
    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, f"es_{env_name}_best.pkl")
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'params': best_params,
            'fitness': best_overall_fitness,
            'config': config,
        }, f)
    print(f"\nSaved checkpoint: {checkpoint_path}")
    
    # Save training metrics
    metrics = {
        'best_fitness': best_overall_fitness,
        'total_time': total_time,
        'num_generations': num_generations,
        'pop_size': pop_size,
        'num_params': num_params,
        **config,
    }
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    
    # Final evaluation with GIFs
    print(f"\nGenerating evaluation GIFs...")
    for eval_idx in range(5):
        key, eval_key = random.split(key)
        frames, reward = rollout_for_gif(
            env, env_params, env_state_init, static_env_params,
            policy, best_params, param_template, episode_length, eval_key
        )
        gif_path = os.path.join(gifs_dir, f"eval_{eval_idx:02d}_reward{int(reward)}.gif")
        imageio.mimsave(gif_path, frames, fps=15, loop=0)
        print(f"  Eval {eval_idx}: reward={reward:.1f}, saved to {gif_path}")
    
    print(f"\n=== Training complete! ===")
    
    wandb.finish()
    
    # Close logger
    sys.stdout = tee_logger.stdout
    tee_logger.close()


if __name__ == "__main__":
    main()
