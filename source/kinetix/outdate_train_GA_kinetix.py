"""
Train SimpleGA on Kinetix environments (non-continual).

Uses the SAME network architecture as PPO (ActorCriticPixelsRNN) for fair comparison.
Uses pixel-based observations by default, multi-discrete action space.
Runs on 20 medium h-tasks (h0-h19).

Usage:
    python train_GA_kinetix.py --env h0_unicycle --gpus 0
    python train_GA_kinetix.py --env h5_angry_birds --gpus 1 --symbolic
"""

import argparse
import os
import sys
import yaml

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
from evosax.algorithms import SimpleGA
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
from kinetix.models import make_network_from_config, ScannedRNN
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
# Environment Setup
# ============================================================================

# Default config for kinetix GA (matching inspiration hyperparams)
KINETIX_CONFIG = {
    "num_generations": 200,  # ~200*20 like inspiration hyperparams
    "pop_size": 256,  # Reduced from 1024 for memory
    "episode_length": 1024,  # env_params.max_timesteps default
    "num_evals": 30,
    "elite_ratio": 0.5,
    "mutation_std": 0.001,  # sigma_init from inspiration
    "init_range": 1.0,  # uniform init in [-init_range, init_range], matches inspiration init_min=-1
    # Network params (same as PPO)
    "fc_layer_depth": 5,
    "fc_layer_width": 128,
    "activation": "tanh",
    "recurrent_model": False,
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


def load_kinetix_env_and_network(env_name, use_pixels=True, fc_layer_depth=5, fc_layer_width=128, 
                                   activation="tanh", recurrent_model=False):
    """Load a kinetix environment and create the PPO network (ActorCriticPixelsRNN)."""
    # Load level from JSON
    level_path = f"m/{env_name}.json"
    env_state, static_env_params, env_params = load_from_json_file(level_path)
    
    # Create config that matches PPO
    obs_type = ObservationType.PIXELS if use_pixels else ObservationType.SYMBOLIC_FLAT
    config = {
        "observation_type": obs_type,
        "action_type": ActionType.MULTI_DISCRETE,
        "train_level_mode": "list",
        "train_levels_list": [level_path],
        # Network config (same as PPO)
        "fc_layer_depth": fc_layer_depth,
        "fc_layer_width": fc_layer_width,
        "activation": activation,
        "recurrent_model": recurrent_model,
        # Static env params needed for network construction
        "static_env_params": to_state_dict(static_env_params),
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
    
    # Create actor-only network (no critic needed for GA)
    network = make_network_from_config(env, env_params, config, actor_only=True)
    
    return env, env_params, static_env_params, env_state, network, config


def get_flat_params(params):
    flat_params, _ = flatten_util.ravel_pytree(params)
    return flat_params


def unflatten_params(flat_params, param_template):
    _, unravel_fn = flatten_util.ravel_pytree(param_template)
    return unravel_fn(flat_params)


# ============================================================================
# Scoring Function (with optional RNN state handling)
# ============================================================================

def make_scoring_fn(env, env_params, env_state_init, network, param_template, episode_length, 
                    recurrent_model=False, num_evals=1):
    """Create JIT-compiled scoring function for kinetix.
    
    When recurrent_model=False, we avoid unnecessary RNN state tracking for efficiency.
    """
    
    # Initialize hidden state once (used as constant when recurrent=False)
    init_hidden = ScannedRNN.initialize_carry(1, 256)
    init_dones = jnp.zeros((1,), dtype=jnp.bool_)
    
    def evaluate_single(flat_params, eval_key):
        params = unflatten_params(flat_params, param_template)
        reset_key, rollout_key = random.split(eval_key)
        
        # Reset environment
        obs, state = env.reset(reset_key, env_params, override_reset_state=env_state_init)
        
        if recurrent_model:
            # Full RNN handling with state tracking
            def step_fn(carry, _):
                obs, state, hidden, dones, total_reward, done_flag, key = carry
                
                obs_batched = jax.tree.map(lambda x: x[jnp.newaxis, jnp.newaxis, ...], obs)
                dones_batched = dones[jnp.newaxis, :]
                ac_in = (obs_batched, dones_batched)
                
                new_hidden, pi = network.apply(params, hidden, ac_in)
                
                key, action_key = random.split(key)
                action = pi.sample(seed=action_key)
                action = action[0, 0, :]
                
                key, step_key = random.split(key)
                next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)
                
                total_reward = total_reward + reward * (1.0 - done_flag)
                done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
                
                # Reset RNN state on done
                new_hidden = jax.lax.cond(
                    done,
                    lambda _: ScannedRNN.initialize_carry(1, 256),
                    lambda _: new_hidden,
                    None
                )
                new_dones = jnp.array([done], dtype=jnp.bool_)
                
                return (next_obs, next_state, new_hidden, new_dones, total_reward, done_flag, key), None
            
            initial_carry = (obs, state, init_hidden, init_dones, 0.0, 0.0, rollout_key)
            (_, _, _, _, total_reward, _, _), _ = jax.lax.scan(
                step_fn, initial_carry, None, length=episode_length
            )
        else:
            # Simplified non-recurrent: no hidden state tracking needed
            def step_fn(carry, _):
                obs, state, total_reward, done_flag, key = carry
                
                obs_batched = jax.tree.map(lambda x: x[jnp.newaxis, jnp.newaxis, ...], obs)
                dones_batched = init_dones[jnp.newaxis, :]
                ac_in = (obs_batched, dones_batched)
                
                # Hidden state is passed but not used when recurrent=False
                _, pi = network.apply(params, init_hidden, ac_in)
                
                key, action_key = random.split(key)
                action = pi.sample(seed=action_key)
                action = action[0, 0, :]
                
                key, step_key = random.split(key)
                next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)
                
                total_reward = total_reward + reward * (1.0 - done_flag)
                done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
                
                return (next_obs, next_state, total_reward, done_flag, key), None
            
            initial_carry = (obs, state, 0.0, 0.0, rollout_key)
            (_, _, total_reward, _, _), _ = jax.lax.scan(
                step_fn, initial_carry, None, length=episode_length
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


def rollout_for_gif(env, env_params, env_state_init, static_env_params, network, 
                    flat_params, param_template, episode_length, key, recurrent_model=False):
    """Rollout for evaluation/GIF generation."""
    params = unflatten_params(flat_params, param_template)
    
    reset_key, rollout_key = random.split(key)
    obs, state = env.reset(reset_key, env_params, override_reset_state=env_state_init)
    
    # Initialize RNN state
    hidden = ScannedRNN.initialize_carry(1, 256)
    dones = jnp.zeros((1,), dtype=jnp.bool_)
    
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
        
        # Format inputs for network
        obs_batched = jax.tree.map(lambda x: x[jnp.newaxis, jnp.newaxis, ...], obs)
        dones_batched = dones[jnp.newaxis, :]
        ac_in = (obs_batched, dones_batched)
        
        # Get action from actor-only network (sample for stochastic evaluation)
        rollout_key, action_key = random.split(rollout_key)
        hidden, pi = network.apply(params, hidden, ac_in)
        action = pi.sample(seed=action_key)[0, 0, :]
        
        rollout_key, step_key = random.split(rollout_key)
        obs, state, reward, done, _ = env.step(step_key, state, action, env_params)
        total_reward += float(reward)
        
        # Update dones for next step
        dones = jnp.array([done], dtype=jnp.bool_)
        
        # Reset RNN on done
        if bool(done):
            hidden = ScannedRNN.initialize_carry(1, 256)
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
    parser = argparse.ArgumentParser(description='SimpleGA on Kinetix (Non-Continual)')
    parser.add_argument('--env', type=str, default='h0_unicycle',
                        choices=ENVIRONMENTS)
    parser.add_argument('--symbolic', action='store_true',
                        help='Use symbolic observations instead of pixels')
    parser.add_argument('--num_generations', type=int, default=None)
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--elite_ratio', type=float, default=None)
    parser.add_argument('--mutation_std', type=float, default=None)
    parser.add_argument('--init_range', type=float, default=None,
                        help='Init population uniform in [-init_range, init_range]')
    parser.add_argument('--num_evals', type=int, default=None)
    parser.add_argument('--fc_layer_depth', type=int, default=None)
    parser.add_argument('--fc_layer_width', type=int, default=None)
    parser.add_argument('--recurrent', action='store_true', help='Use recurrent model (GRU)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--project_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='Kinetix-GA-noncontinual')
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
    episode_length = cfg["episode_length"]
    num_evals = args.num_evals or cfg["num_evals"]
    elite_ratio = args.elite_ratio or cfg["elite_ratio"]
    mutation_std = args.mutation_std or cfg["mutation_std"]
    init_range = args.init_range or cfg["init_range"]
    fc_layer_depth = args.fc_layer_depth or cfg["fc_layer_depth"]
    fc_layer_width = args.fc_layer_width or cfg["fc_layer_width"]
    activation = cfg["activation"]
    recurrent_model = args.recurrent or cfg["recurrent_model"]
    
    # Output directory
    project_dir = args.project_dir or os.path.join(REPO_ROOT, "projects", "kinetix")
    output_dir = args.output_dir or os.path.join(project_dir, "vanilla", "ga", env_name, f"trial_{trial}")
    os.makedirs(output_dir, exist_ok=True)
    gifs_dir = os.path.join(output_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, "train.log")
    tee_logger = Tee(log_file)
    sys.stdout = tee_logger
    
    print("=" * 60)
    print(f"SimpleGA on Kinetix: {env_name} (Non-Continual)")
    print("=" * 60)
    print(f"  Observation type: {obs_type}")
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}")
    print(f"  Elite ratio: {elite_ratio}, Mutation std: {mutation_std}, Init range: {init_range}")
    print(f"  Num evals: {num_evals}")
    print(f"  Network: fc_depth={fc_layer_depth}, fc_width={fc_layer_width}, "
          f"activation={activation}, recurrent={recurrent_model}")
    print(f"  Output dir: {output_dir}")
    
    key = random.key(seed)
    
    # Load kinetix environment and create network (same as PPO)
    print(f"\nLoading kinetix environment: {env_name}...")
    env, env_params, static_env_params, env_state_init, network, config = load_kinetix_env_and_network(
        env_name, 
        use_pixels=use_pixels,
        fc_layer_depth=fc_layer_depth,
        fc_layer_width=fc_layer_width,
        activation=activation,
        recurrent_model=recurrent_model,
    )
    print(f"  Environment loaded successfully!")
    print(f"  Network: ActorCriticPixelsRNN (same as PPO)")
    
    # Initialize network parameters
    key, init_key, reset_key = random.split(key, 3)
    dummy_obs, _ = env.reset(reset_key, env_params, override_reset_state=env_state_init)
    
    # Initialize with proper input shapes
    dummy_hidden = ScannedRNN.initialize_carry(1, 256)
    dummy_obs_batched = jax.tree.map(lambda x: x[jnp.newaxis, jnp.newaxis, ...], dummy_obs)
    dummy_dones = jnp.zeros((1, 1), dtype=jnp.bool_)
    dummy_ac_in = (dummy_obs_batched, dummy_dones)
    
    param_template = network.init(init_key, dummy_hidden, dummy_ac_in)
    flat_params = get_flat_params(param_template)
    num_params = flat_params.shape[0]
    print(f"  Parameters: {num_params:,}")
    
    # Save info.yaml with parameter count
    info_path = os.path.join(output_dir, "info.yaml")
    info_data = {
        'num_params': int(num_params),
        'network': {
            'fc_layer_depth': fc_layer_depth,
            'fc_layer_width': fc_layer_width,
            'activation': activation,
            'recurrent': recurrent_model,
            'obs_type': str(obs_type),
        },
        'env_name': env_name,
    }
    with open(info_path, 'w') as f:
        yaml.dump(info_data, f, default_flow_style=False)
    print(f"  Saved info to: {info_path}")
    
    # Create scoring function
    print(f"\nCreating scoring function...")
    scoring_fn = make_scoring_fn(
        env, env_params, env_state_init, network,
        param_template, episode_length, recurrent_model, num_evals
    )
    
    # Initialize wandb
    config_log = {
        'env': env_name, 'obs_type': obs_type,
        'num_generations': num_generations, 'pop_size': pop_size,
        'seed': seed, 'trial': trial,
        'elite_ratio': elite_ratio, 'mutation_std': mutation_std,
        'init_range': init_range, 'num_evals': num_evals, 
        'fc_layer_depth': fc_layer_depth, 'fc_layer_width': fc_layer_width,
        'activation': activation, 'recurrent_model': recurrent_model,
        'num_params': num_params,
        'network_type': 'ActorCriticPixelsRNN',
    }
    wandb.init(
        project=args.wandb_project, config=config_log,
        name=f"ga_{env_name}_{obs_type}_trial{trial}", reinit=True
    )
    
    # Initialize GA with multi-GPU sharding
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
    std_schedule = lambda gen: mutation_std
    
    ga = SimpleGA(
        population_size=pop_size,
        solution=jnp.zeros(num_params),
        std_schedule=std_schedule,
    )
    ga.elite_ratio = elite_ratio
    ga_params = jax.device_put(ga.default_params, replicate_sharding)
    
    key, init_key = random.split(key)
    # Uniform initialization in [-init_range, init_range] (matching inspiration)
    init_population = random.uniform(init_key, (pop_size, num_params), minval=-init_range, maxval=init_range)
    init_fitness = jnp.full(pop_size, jnp.inf)
    
    key, state_init_key = random.split(key)
    ga_state = jax.jit(ga.init, out_shardings=replicate_sharding)(
        state_init_key, init_population, init_fitness, ga_params
    )
    
    # JIT compile ask/tell
    @jax.jit
    def jit_ask(key, state, params):
        population, new_state = ga.ask(key, state, params)
        population = jax.device_put(population, parallel_sharding)
        return population, new_state
    
    @jax.jit
    def jit_tell(key, population, fitness, state, params):
        return ga.tell(key, population, fitness, state, params)
    
    # Warmup JIT
    print("\nJIT compiling...")
    key, warmup_key, warmup_ask_key = random.split(key, 3)
    warmup_pop, _ = jit_ask(warmup_ask_key, ga_state, ga_params)
    _, _ = scoring_fn(warmup_pop, warmup_key)
    print("  JIT compilation complete!")
    
    # Training loop
    best_overall_fitness = -float('inf')
    best_params = None
    start_time = time.time()
    
    print(f"\nStarting training...")
    
    for gen in range(num_generations):
        key, ask_key, eval_key, tell_key = random.split(key, 4)
        
        population, ga_state = jit_ask(ask_key, ga_state, ga_params)
        fitness, mean_fitness = scoring_fn(population, eval_key)
        
        mean_fitness_host = jax.device_get(mean_fitness)
        population_host = jax.device_get(population)
        
        # evosax minimizes, so negate fitness for maximization
        ga_state, _ = jit_tell(tell_key, population, -fitness, ga_state, ga_params)
        
        # Track using mean fitness
        gen_best = float(np.max(mean_fitness_host))
        gen_mean = float(np.mean(mean_fitness_host))
        best_idx = int(np.argmax(mean_fitness_host))
        
        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best
            best_params = population_host[best_idx].copy()
        
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
    
    # Use best tracked params
    if best_params is None:
        best_params = jax.device_get(ga_state.best_member)
    
    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, f"ga_{env_name}_best.pkl")
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'params': best_params,
            'fitness': best_overall_fitness,
            'config': config_log,
        }, f)
    print(f"\nSaved checkpoint: {checkpoint_path}")
    
    # Save training metrics
    metrics = {
        'best_fitness': best_overall_fitness,
        'total_time': total_time,
        'num_generations': num_generations,
        'pop_size': pop_size,
        'num_params': num_params,
        **config_log,
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
            env, env_params, env_state_init, static_env_params, network,
            best_params, param_template, episode_length, eval_key, recurrent_model
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
