"""Continual GA (SimpleGA_Elitist) training on Kinetix environments using evosax.

Trains sequentially through all 20 medium h-tasks WITHOUT resetting the
population between tasks.  Each task receives ``--generations_per_task``
generations (default 200).  The ES state (population + elite archive) carries over.

Uses the actor-only network (ActorOnlyPixelsRNN) - no critic.

Usage:
    python experiments/ga_continual.py --gpu 0
    python experiments/ga_continual.py --gpu 1 --generations_per_task 100
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import NamedTuple

# ── GPU selection BEFORE jax import ──────────────────────────────
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if "--gpu" in sys.argv:
    _gpu_idx = sys.argv.index("--gpu")
    if _gpu_idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_gpu_idx + 1]
        print(f"CUDA_VISIBLE_DEVICES={sys.argv[_gpu_idx + 1]}")

import imageio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import yaml

from flax.serialization import to_state_dict

import wandb

# ── Kinetix imports ──────────────────────────────────────────────
from kinetix.environment import make_reset_fn_from_config
from kinetix.environment.env import make_kinetix_env
from kinetix.models import ScannedRNN, make_network_from_config
from kinetix.render.renderer_pixels import make_render_pixels
from kinetix.util import normalise_config
from kinetix.util.saving import load_from_json_file

# ── evosax ───────────────────────────────────────────────────────
from evosax.algorithms.population_based.simple_ga import SimpleGA
from simple_ga_elitist import SimpleGA_Elitist

# ── Constants ────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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


# ── Tee helper ───────────────────────────────────────────────────

class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


# ── ParameterReshaper ────────────────────────────────────────────

class ParameterReshaper:
    """Flat-vector <-> pytree conversion using jax.flatten_util."""

    def __init__(self, params):
        flat, self._unravel_fn = jax.flatten_util.ravel_pytree(params)
        self.total_params = flat.shape[0]

    def reshape_single(self, flat_params):
        return self._unravel_fn(flat_params)

    def reshape(self, batch_flat_params):
        return jax.vmap(self._unravel_fn)(batch_flat_params)

    def flatten_single(self, params):
        flat, _ = jax.flatten_util.ravel_pytree(params)
        return flat


# ── Load the base config yaml ───────────────────────────────────

def load_base_config():
    yaml_candidates = [
        os.path.join(REPO_ROOT, "ga_example", "kinetix_config_pixels.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "kinetix_config_pixels.yaml"),
    ]
    for p in yaml_candidates:
        if os.path.exists(p):
            with open(p, "r") as f:
                return yaml.load(f, Loader=yaml.SafeLoader)
    raise FileNotFoundError(
        "Cannot find kinetix_config_pixels.yaml. Looked in: " + ", ".join(yaml_candidates)
    )


def load_env(env_name, base_yaml_config):
    """Build env objects for a single task name (e.g. 'h0_unicycle')."""
    config = {**base_yaml_config}
    config.setdefault("seed", 0)
    config = normalise_config(config, name="PPO")

    qualified = env_name if env_name.startswith(("m/", "s/", "l/")) else f"m/{env_name}"
    env_state, static_ep, ep = load_from_json_file(qualified)

    config["env_params"] = to_state_dict(ep)
    config["static_env_params"] = to_state_dict(static_ep)

    reset_fn = make_reset_fn_from_config(config, ep, static_ep)
    env = make_kinetix_env(
        observation_type=config["observation_type"],
        action_type=config["action_type"],
        reset_fn=reset_fn,
        env_params=ep,
        static_env_params=static_ep,
    )
    return config, env, env_state, static_ep, ep


# ── Core continual training function ────────────────────────────

def train_ga_continual(
    *,
    popsize: int = 1024,
    generations_per_task: int = 200,
    sigma_init: float = 0.001,
    seed: int = 0,
    trial_idx: int = 1,
    project_dir: str | None = None,
    use_wandb: bool = True,
    wandb_project: str = "Kinetix-continual-ga",
    episode_length: int = 1000,
    eval_reps: int = 3,
    evolve_reps: int = 3,
    eval_batch_size: int = 128,
    optimizer: str = "SimpleGA_Elitist",
):
    total_reps = max(eval_reps, evolve_reps)
    rng = jr.PRNGKey(seed)

    # ── load all 20 envs up-front ─────────────────────────────────
    base_yaml = load_base_config()
    print("Loading all environments...")
    envs_data = []
    for ename in ENVIRONMENTS:
        print(f"  Loading {ename}...")
        envs_data.append(load_env(ename, base_yaml))
    print(f"  All {len(ENVIRONMENTS)} environments loaded.\n")

    # Use first env to initialise network & reshaper (same architecture)
    config0, env0, init_es0, static_ep0, ep0 = envs_data[0]
    network = make_network_from_config(env0, ep0, config0, actor_only=True)

    rng, init_rng = jr.split(rng)
    dummy_obs, _ = jax.vmap(env0.reset, (0, None))(jr.split(init_rng, 1), ep0)
    dones = jnp.zeros(1, dtype=jnp.bool_)
    init_hstate = ScannedRNN.initialize_carry(1)
    init_x = jax.tree.map(lambda x: x[None, ...], (dummy_obs, dones))

    rng, param_rng = jr.split(rng)
    network_params = network.init(param_rng, init_hstate, init_x)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(network_params))
    print(f"Actor-only network param count: {param_count}")

    reshaper = ParameterReshaper(network_params)
    num_dims = reshaper.total_params
    print(f"Total flat params: {num_dims}")

    total_generations = generations_per_task * len(ENVIRONMENTS)

    # ── project dir setup ─────────────────────────────────────────
    output_dir = None
    tee_logger = None
    if project_dir:
        output_dir = os.path.join(
            project_dir, "continual", "ga", "all_tasks", f"trial_{trial_idx}"
        )
        os.makedirs(output_dir, exist_ok=True)
        gifs_dir = os.path.join(output_dir, "gifs")
        os.makedirs(gifs_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "train.log")
        tee_logger = Tee(log_file)
        sys.stdout = tee_logger

    print(f"\n=== Kinetix GA Continual Training ===")
    print(f"  Trial: {trial_idx}")
    print(f"  Seed: {seed}")
    print(f"  Population size: {popsize}")
    print(f"  Generations per task: {generations_per_task}")
    print(f"  Total generations: {total_generations}")
    print(f"  Sigma init: {sigma_init}")
    print(f"  Param count: {param_count}")
    print(f"  Eval reps (reporting): {eval_reps}")
    print(f"  Evolve reps (selection): {evolve_reps}")
    print(f"  Num tasks: {len(ENVIRONMENTS)}")
    if output_dir:
        print(f"  Output directory: {output_dir}")
    print(f"======================================\n")

    # ── wandb ─────────────────────────────────────────────────────
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=f"GA_continual_trial{trial_idx}_seed{seed}",
            config={
                "popsize": popsize,
                "generations_per_task": generations_per_task,
                "total_generations": total_generations,
                "sigma_init": sigma_init,
                "seed": seed,
                "trial_idx": trial_idx,
                "param_count": param_count,
                "episode_length": episode_length,
                "eval_reps": eval_reps,
                "evolve_reps": evolve_reps,
                "optimizer": optimizer,
                "num_tasks": len(ENVIRONMENTS),
                "continual": True,
            },
        )

    # ── evosax strategy ───────────────────────────────────────────
    StrategyClass = SimpleGA_Elitist if optimizer == "SimpleGA_Elitist" else SimpleGA
    print(f"  Optimizer: {optimizer}")
    strategy = StrategyClass(
        population_size=popsize,
        solution=jnp.zeros(num_dims),
        std_schedule=optax.constant_schedule(sigma_init),
    )
    es_params = strategy.default_params

    rng, pop_rng = jr.split(rng)
    init_population = jr.uniform(pop_rng, (popsize, num_dims), minval=-1.0, maxval=1.0)
    init_fitness = jnp.full(popsize, jnp.inf)

    rng, init_rng = jr.split(rng)
    es_state = strategy.init(init_rng, init_population, init_fitness, es_params)

    # ── helper: build jitted rollout & eval for a given env ───────

    def make_rollout_and_eval(env, ep, init_env_state):
        """Return a jitted eval_population fn for one environment."""

        @jax.jit
        def rollout_single(params, rng_key):
            rng_key, reset_key, step_key = jr.split(rng_key, 3)
            obs, env_state = env.reset(
                reset_key, env_params=ep, override_reset_state=init_env_state
            )
            dones = jnp.zeros(1, dtype=jnp.bool_)
            hstate = ScannedRNN.initialize_carry(1)
            init_carry = (env_state, obs, dones, hstate, step_key)

            def _step(carry, _):
                env_st, obs_, done_, hstate_, rng_ = carry
                rng_, act_rng, step_rng = jr.split(rng_, 3)
                ac_in = jax.tree.map(
                    lambda x: x[None, None, ...], (obs_, done_),
                )
                new_hstate, pi = network.apply(params, hstate_, ac_in)
                action = pi.sample(seed=act_rng)[0, 0, :]
                obs_next, env_st_next, reward, done, info = env.step(
                    step_rng, state=env_st, action=action, env_params=ep
                )
                done_ = jnp.expand_dims(done, axis=0)
                return (env_st_next, obs_next, done_, new_hstate, rng_), (reward, done)

            _, (rewards, dones_seq) = jax.lax.scan(
                _step, init_carry, None, length=episode_length
            )
            any_done = jnp.any(dones_seq)
            first_done = jnp.argmax(dones_seq)
            first_done = jnp.where(any_done, first_done, episode_length)
            idxs = jnp.arange(episode_length)
            rewards = jnp.where(idxs > first_done, 0.0, rewards)
            return jnp.sum(rewards), first_done

        @jax.jit
        def _eval_batch(flat_batch, rng_key):
            params_batch = reshaper.reshape(flat_batch)
            rep_keys = jr.split(rng_key, total_reps)

            def _eval_one_rep(rep_key):
                return jax.vmap(rollout_single, in_axes=(0, None))(params_batch, rep_key)

            all_fit, all_len = jax.vmap(_eval_one_rep)(rep_keys)
            return jnp.transpose(all_fit), jnp.transpose(all_len)

        def eval_population(flat_pop, rng_key):
            """Evaluate population in chunks of eval_batch_size to avoid OOM."""
            n = flat_pop.shape[0]
            fit_parts, len_parts = [], []
            for start in range(0, n, eval_batch_size):
                batch = flat_pop[start : start + eval_batch_size]
                rng_key, batch_key = jr.split(rng_key)
                f, l = _eval_batch(batch, batch_key)
                jax.block_until_ready(f)
                fit_parts.append(f)
                len_parts.append(l)
            return jnp.concatenate(fit_parts, axis=0), jnp.concatenate(len_parts, axis=0)

        return eval_population, rollout_single

    # ── helper: generate GIFs for one task ────────────────────────

    def save_task_gifs(task_idx, env_name, config, env, ep, static_ep,
                       init_env_state, best_flat_params, num_gifs=10):
        if output_dir is None:
            return
        task_gifs = os.path.join(output_dir, "gifs", f"task{task_idx}_{env_name}")
        os.makedirs(task_gifs, exist_ok=True)

        eval_env = make_kinetix_env(
            observation_type=config["observation_type"],
            action_type=config["action_type"],
            reset_fn=make_reset_fn_from_config(config, ep, static_ep),
            static_env_params=static_ep,
        )
        render_sep = eval_env.static_env_params.replace(downscale=4)
        pixel_renderer = jax.jit(make_render_pixels(ep, render_sep))
        best_params_tree = reshaper.reshape_single(best_flat_params)

        @jax.jit
        def get_action(params, hstate, obs_batched, done, rng):
            ac_in = jax.tree.map(lambda x: x[None, ...], (obs_batched, done))
            new_hstate, pi = network.apply(params, hstate, ac_in)
            action = pi.sample(seed=rng).squeeze(0)
            return action, new_hstate

        @jax.jit
        def env_step_jit(rng, state, action):
            return eval_env.step(rng, state, action, ep)

        @jax.jit
        def env_reset_jit(rng):
            return eval_env.reset(rng, ep, override_reset_state=init_env_state)

        for gif_idx in range(num_gifs):
            eval_rng = jr.PRNGKey(seed * 1000 + task_idx * 100 + gif_idx)
            obs, env_state = env_reset_jit(eval_rng)
            hstate = ScannedRNN.initialize_carry(1)
            done = jnp.zeros(1, dtype=jnp.bool_)
            frames = []
            total_reward = 0.0

            for step in range(ep.max_timesteps):
                frame = np.array(pixel_renderer(env_state))
                frame = frame.transpose(1, 0, 2)[::-1].astype(np.uint8)
                frames.append(frame)

                eval_rng, act_rng = jr.split(eval_rng)
                obs_batched = jax.tree.map(lambda x: x[None, ...], obs)
                action, hstate = get_action(
                    best_params_tree, hstate, obs_batched, done, act_rng
                )
                action = action.squeeze(0) if hasattr(action, "squeeze") else action[0]

                eval_rng, step_rng = jr.split(eval_rng)
                obs, env_state, reward, step_done, info = env_step_jit(
                    step_rng, env_state, action
                )
                total_reward += float(reward)
                if bool(step_done):
                    frame = np.array(pixel_renderer(env_state))
                    frame = frame.transpose(1, 0, 2)[::-1].astype(np.uint8)
                    frames.append(frame)
                    break

            gif_path = os.path.join(
                task_gifs, f"rollout_{gif_idx:02d}_reward{int(total_reward)}.gif"
            )
            imageio.mimsave(gif_path, frames, fps=15, loop=0)
        print(f"    Saved {num_gifs} GIFs for task {task_idx} ({env_name}) → {task_gifs}")

    # ── training loop across tasks ────────────────────────────────
    print("Starting continual GA training loop...")
    start_time = time.time()
    global_gen = 0
    best_fitness_ever = -jnp.inf
    best_params_ever = None

    for task_idx, env_name in enumerate(ENVIRONMENTS):
        config_t, env_t, init_es_t, static_ep_t, ep_t = envs_data[task_idx]
        eval_pop_fn, _ = make_rollout_and_eval(env_t, ep_t, init_es_t)

        task_best_fitness = -jnp.inf
        task_best_params = None
        bare = env_name.split("/")[-1] if "/" in env_name else env_name

        print(f"\n{'='*60}")
        print(f"Task {task_idx}/{len(ENVIRONMENTS)-1}: {bare}  "
              f"(gens {global_gen}..{global_gen + generations_per_task - 1})")
        print(f"{'='*60}")

        for local_gen in range(generations_per_task):
            rng, ask_key, eval_key, tell_key = jr.split(rng, 4)

            flat_pop, es_state = strategy.ask(ask_key, es_state, es_params)
            all_fitnesses, all_ep_lengths = eval_pop_fn(flat_pop, eval_key)
            jax.block_until_ready(all_fitnesses)

            evolve_fit = jnp.mean(all_fitnesses[:, :evolve_reps], axis=1)
            report_fit = jnp.mean(all_fitnesses[:, :eval_reps], axis=1)
            mean_ep_lengths = jnp.mean(all_ep_lengths[:, :eval_reps], axis=1)

            report_np = np.array(report_fit)
            evolve_np = np.array(evolve_fit)
            best_report_idx = int(np.argmax(report_np))
            best_evolve_idx = int(np.argmax(evolve_np))

            # Best individual's per-rollout stats
            best_rollouts = np.array(all_fitnesses[best_report_idx, :eval_reps])
            rep_best_mean = float(np.mean(best_rollouts))
            rep_best_min  = float(np.min(best_rollouts))
            rep_best_max  = float(np.max(best_rollouts))
            evo_best = float(evolve_np[best_evolve_idx])
            pop_mean = float(np.mean(report_np))
            mean_ep_len = float(np.mean(np.array(mean_ep_lengths)))

            if rep_best_mean > task_best_fitness:
                task_best_fitness = rep_best_mean
                task_best_params = flat_pop[best_report_idx]

            if rep_best_mean > best_fitness_ever:
                best_fitness_ever = rep_best_mean
                best_params_ever = flat_pop[best_report_idx]

            elapsed = time.time() - start_time
            print(
                f"Gen {global_gen:4d} (task {task_idx} local {local_gen:3d})  "
                f"best(mean={rep_best_mean:8.2f} min={rep_best_min:8.2f} max={rep_best_max:8.2f})  "
                f"pop_mean={pop_mean:8.2f}  "
                f"ep_len={mean_ep_len:6.0f}  task_best={task_best_fitness:8.2f}  "
                f"time={elapsed:6.1f}s"
            )

            if use_wandb:
                wandb.log(
                    {
                        "generation": global_gen,
                        "task": task_idx,
                        "task_name": bare,
                        "best/mean": rep_best_mean,
                        "best/min": rep_best_min,
                        "best/max": rep_best_max,
                        "best/task_best": float(task_best_fitness),
                        "best/best_ever": float(best_fitness_ever),
                        "population/mean": pop_mean,
                        "episode_length/mean": mean_ep_len,
                    },
                    step=global_gen,
                )

            # tell – evosax minimises, negate evolve fitness
            es_state, _ = strategy.tell(
                tell_key, flat_pop, -evolve_fit, es_state, es_params
            )
            global_gen += 1

        # ── end of task: save GIFs ────────────────────────────────
        if task_best_params is not None:
            print(f"  Task {task_idx} ({bare}) finished.  task_best={task_best_fitness:.2f}")
            save_task_gifs(
                task_idx, bare, config_t, env_t, ep_t, static_ep_t,
                init_es_t, task_best_params,
            )
        else:
            print(f"  Task {task_idx} ({bare}) finished.  No best params found.")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Continual training complete! Total time: {total_time:.1f}s")
    print(f"Best fitness ever: {best_fitness_ever:.2f}")
    print(f"{'='*60}")

    # ── save checkpoint ───────────────────────────────────────────
    if output_dir and best_params_ever is not None:
        best_tree = reshaper.reshape_single(best_params_ever)
        ckpt_path = os.path.join(output_dir, "ga_continual_best.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump(
                {
                    "params": jax.tree.map(np.array, best_tree),
                    "best_fitness": float(best_fitness_ever),
                    "total_generations": total_generations,
                    "generations_per_task": generations_per_task,
                    "seed": seed,
                    "popsize": popsize,
                    "sigma_init": sigma_init,
                },
                f,
            )
        print(f"  Saved checkpoint: {ckpt_path}")

        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "best_fitness": float(best_fitness_ever),
                    "total_time_s": total_time,
                    "total_generations": total_generations,
                    "generations_per_task": generations_per_task,
                    "popsize": popsize,
                    "sigma_init": sigma_init,
                    "seed": seed,
                    "trial_idx": trial_idx,
                    "param_count": param_count,
                    "num_tasks": len(ENVIRONMENTS),
                    "eval_reps": eval_reps,
                    "evolve_reps": evolve_reps,
                },
                f,
                indent=2,
            )
        print(f"  Saved metrics: {metrics_path}")

    # ── cleanup ───────────────────────────────────────────────────
    if use_wandb:
        wandb.finish()
    if tee_logger:
        sys.stdout = tee_logger.stdout
        tee_logger.close()

    return best_fitness_ever


# ── CLI ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Continual GA on Kinetix - sequential tasks, no population reset"
    )
    parser.add_argument("--gpu", type=str, default=None, help="GPU device ID")
    parser.add_argument("--popsize", type=int, default=1024)
    parser.add_argument("--generations_per_task", type=int, default=200)
    parser.add_argument("--sigma_init", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument(
        "--eval_batch_size", type=int, default=128,
        help="Chunk size for batched population evaluation (default 128, reduce if OOM)",
    )
    parser.add_argument(
        "--eval_reps", type=int, default=3,
        help="Number of rollouts averaged for reported fitness (default 3)",
    )
    parser.add_argument(
        "--evolve_reps", type=int, default=3,
        help="Number of rollouts averaged for evolution fitness / tell() (default 3)",
    )
    parser.add_argument(
        "--optimizer", type=str, default="SimpleGA_Elitist",
        choices=["SimpleGA", "SimpleGA_Elitist"],
        help="Evolution strategy: SimpleGA or SimpleGA_Elitist (default: SimpleGA_Elitist)",
    )
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Kinetix-continual-ga")
    parser.add_argument(
        "--project_dir", type=str, default=None,
        help="Root project dir for structured output (e.g. projects/kinetix)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # GPU already set before JAX import (top of file)

    for trial in range(1, args.num_trials + 1):
        print(f"\n{'#'*60}")
        print(f"# Continual GA  Trial: {trial}")
        print(f"{'#'*60}")
        train_ga_continual(
            popsize=args.popsize,
            generations_per_task=args.generations_per_task,
            sigma_init=args.sigma_init,
            seed=args.seed + trial - 1,
            trial_idx=trial,
            project_dir=args.project_dir,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            episode_length=args.episode_length,
            eval_reps=args.eval_reps,
            evolve_reps=args.evolve_reps,
            eval_batch_size=args.eval_batch_size,
            optimizer=args.optimizer,
        )


if __name__ == "__main__":
    main()
