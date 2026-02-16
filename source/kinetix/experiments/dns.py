"""DNS (Dominated Novelty Search) training on Kinetix environments.

Custom DNS implementation (no QDax dependency). Uses the actor-only network
(ActorOnlyPixelsRNN) with isoline variation, novelty-based selection, and
fitness+novelty combined scoring.

Usage:
    python experiments/dns.py --env h0_unicycle --gpu 0
    python experiments/dns.py --env h5_angry_birds --gpu 1 --pop_size 512
    python experiments/dns.py                              # all 20 envs
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import NamedTuple

# -- GPU selection BEFORE jax import ------------------------------------------
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
import yaml

from flax.serialization import to_state_dict

import wandb

# -- Kinetix imports ----------------------------------------------------------
from kinetix.environment import make_reset_fn_from_config
from kinetix.environment.env import make_kinetix_env
from kinetix.models import ScannedRNN, make_network_from_config
from kinetix.render.renderer_pixels import make_render_pixels
from kinetix.util import normalise_config
from kinetix.util.saving import load_from_json_file

# -- Constants ----------------------------------------------------------------

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

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


# -- Logging helper -----------------------------------------------------------

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


# -- ParameterReshaper --------------------------------------------------------

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


# -- DNS algorithm ------------------------------------------------------------

def compute_novelty(descriptors: jnp.ndarray, k: int = 3) -> jnp.ndarray:
    """Novelty = mean distance to k nearest neighbors in descriptor space."""
    descriptors = jnp.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
    diffs = descriptors[:, None, :] - descriptors[None, :, :]
    distances = jnp.linalg.norm(diffs, axis=-1)
    # Exclude self-distance
    distances = distances + jnp.eye(distances.shape[0]) * 1e10
    k_nearest = jnp.sort(distances, axis=1)[:, :k]
    novelty = jnp.mean(k_nearest, axis=1)
    return jnp.nan_to_num(novelty, nan=0.0, posinf=0.0, neginf=0.0)


def dns_selection(
    genotypes, fitnesses, descriptors,
    new_genotypes, new_fitnesses, new_descriptors,
    population_size, k,
):
    """DNS selection: combine parents + offspring, rank by normalised fitness + novelty."""
    combined_geno = jnp.concatenate([genotypes, new_genotypes], axis=0)
    combined_fit = jnp.concatenate([fitnesses, new_fitnesses], axis=0)
    combined_desc = jnp.concatenate([descriptors, new_descriptors], axis=0)

    combined_nov = compute_novelty(combined_desc, k)
    combined_fit = jnp.nan_to_num(combined_fit, nan=-1e6, posinf=1e6, neginf=-1e6)

    # Normalise fitness to [0, 1]
    f_min, f_max = jnp.min(combined_fit), jnp.max(combined_fit)
    f_range = jnp.maximum(f_max - f_min, 1e-8)
    norm_fit = (combined_fit - f_min) / f_range

    # Normalise novelty to [0, 1]
    n_min, n_max = jnp.min(combined_nov), jnp.max(combined_nov)
    n_range = jnp.maximum(n_max - n_min, 1e-8)
    norm_nov = (combined_nov - n_min) / n_range

    scores = jnp.nan_to_num(norm_fit + norm_nov, nan=-1e6)
    sel_idx = jnp.argsort(scores)[-population_size:]

    return (
        combined_geno[sel_idx],
        combined_fit[sel_idx],
        combined_desc[sel_idx],
        combined_nov[sel_idx],
    )


def isoline_variation(genotypes, key, iso_sigma=0.001, line_sigma=0.0, batch_size=256):
    """Isoline crossover + Gaussian noise to generate offspring."""
    pop_size = genotypes.shape[0]
    param_dim = genotypes.shape[1]

    key, k1, k2, k3, k4 = jr.split(key, 5)
    p1_idx = jr.randint(k1, (batch_size,), 0, pop_size)
    p2_idx = jr.randint(k2, (batch_size,), 0, pop_size)

    p1 = genotypes[p1_idx]
    p2 = genotypes[p2_idx]

    alpha = jr.uniform(k3, (batch_size, param_dim), minval=-line_sigma, maxval=1.0 + line_sigma)
    offspring = p1 + alpha * (p2 - p1)
    offspring = offspring + jr.normal(k4, (batch_size, param_dim)) * iso_sigma

    return offspring


# -- Diversity metrics --------------------------------------------------------

def compute_fitness_diversity(fitnesses):
    return float(jnp.std(fitnesses))


def compute_genomic_diversity(genotypes, sample_size=32):
    pop_size = genotypes.shape[0]
    if pop_size < 2:
        return 0.0
    actual = min(sample_size, pop_size)
    indices = np.random.choice(pop_size, size=actual, replace=False)
    sampled = np.array(genotypes[indices])
    g_min = np.min(sampled, axis=0, keepdims=True)
    g_max = np.max(sampled, axis=0, keepdims=True)
    g_range = np.maximum(g_max - g_min, 1e-8)
    norm = (sampled - g_min) / g_range
    diffs = norm[:, None, :] - norm[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    mask = np.triu(np.ones((actual, actual)), k=1)
    return float(np.sum(dists * mask) / np.sum(mask))


# -- Build config from YAML --------------------------------------------------

def build_config_for_env(env_name):
    yaml_candidates = [
        os.path.join(REPO_ROOT, "ga_example", "kinetix_config_pixels.yaml"),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "kinetix_config_pixels.yaml",
        ),
    ]
    yaml_path = None
    for p in yaml_candidates:
        if os.path.exists(p):
            yaml_path = p
            break
    if yaml_path is None:
        raise FileNotFoundError(
            "Cannot find kinetix_config_pixels.yaml. Looked in: "
            + ", ".join(yaml_candidates)
        )

    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config.setdefault("seed", 0)
    config = normalise_config(config, name="PPO")

    qualified = env_name if env_name.startswith(("m/", "s/", "l/")) else f"m/{env_name}"
    env_state, static_ep, ep = load_from_json_file(qualified)

    config["env_params"] = to_state_dict(ep)
    config["static_env_params"] = to_state_dict(static_ep)
    config["train_levels_list"] = [f"{qualified}.json"]

    return config, env_state, static_ep, ep


# -- Core training function ---------------------------------------------------

def train_dns(
    env_name: str,
    *,
    pop_size: int = 1024,
    batch_size: int = 512,
    num_generations: int = 200,
    iso_sigma: float = 0.001,
    line_sigma: float = 0.0,
    k: int = 3,
    num_evals: int = 3,
    seed: int = 0,
    trial_idx: int = 1,
    project_dir: str | None = None,
    use_wandb: bool = True,
    wandb_project: str = "Kinetix-noncontinual-dns",
    episode_length: int = 1000,
    eval_batch_size: int = 128,
    log_interval: int = 10,
):
    """
    DNS training loop for a single Kinetix environment.

    1. Build env + actor-only network
    2. Initialise population of flat parameter vectors
    3. Isoline variation -> scoring -> DNS selection loop
    4. Save checkpoint, GIFs, log to wandb
    """
    rng = jr.PRNGKey(seed)

    # -- build env & config ---------------------------------------------------
    config, init_env_state, static_ep, ep = build_config_for_env(env_name)

    reset_fn = make_reset_fn_from_config(config, ep, static_ep)
    env = make_kinetix_env(
        observation_type=config["observation_type"],
        action_type=config["action_type"],
        reset_fn=reset_fn,
        env_params=ep,
        static_env_params=static_ep,
    )

    # -- actor-only network ---------------------------------------------------
    network = make_network_from_config(env, ep, config, actor_only=True)

    rng, init_rng = jr.split(rng)
    dummy_obs, _ = jax.vmap(env.reset, (0, None))(jr.split(init_rng, 1), ep)
    dones = jnp.zeros(1, dtype=jnp.bool_)
    init_hstate = ScannedRNN.initialize_carry(1)
    init_x = jax.tree.map(lambda x: x[None, ...], (dummy_obs, dones))

    rng, param_rng = jr.split(rng)
    network_params = network.init(param_rng, init_hstate, init_x)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(network_params))
    print(f"Actor-only network param count: {param_count}")

    # -- parameter reshaper ---------------------------------------------------
    reshaper = ParameterReshaper(network_params)
    num_dims = reshaper.total_params
    print(f"Total flat params: {num_dims}")

    # -- project dir setup ----------------------------------------------------
    bare_name = env_name.split("/")[-1] if "/" in env_name else env_name
    output_dir = None
    tee_logger = None

    if project_dir:
        output_dir = os.path.join(
            project_dir, "vanilla", "dns", bare_name, f"trial_{trial_idx}"
        )
        os.makedirs(output_dir, exist_ok=True)
        gifs_dir = os.path.join(output_dir, "gifs")
        os.makedirs(gifs_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "train.log")
        tee_logger = Tee(log_file)
        sys.stdout = tee_logger

    print(f"\n=== Kinetix DNS Training ===")
    print(f"  Environment: {env_name}")
    print(f"  Trial: {trial_idx}")
    print(f"  Seed: {seed}")
    print(f"  Population size: {pop_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Generations: {num_generations}")
    print(f"  iso_sigma: {iso_sigma}")
    print(f"  line_sigma: {line_sigma}")
    print(f"  k (novelty neighbors): {k}")
    print(f"  Num evals (reporting): {num_evals}")
    print(f"  Param count: {param_count}")
    if output_dir:
        print(f"  Output directory: {output_dir}")
    print(f"============================\n")

    # -- wandb ----------------------------------------------------------------
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=f"DNS_{bare_name}_trial{trial_idx}_seed{seed}",
            config={
                "env_name": env_name,
                "pop_size": pop_size,
                "batch_size": batch_size,
                "num_generations": num_generations,
                "iso_sigma": iso_sigma,
                "line_sigma": line_sigma,
                "k": k,
                "num_evals": num_evals,
                "seed": seed,
                "trial_idx": trial_idx,
                "param_count": param_count,
                "episode_length": episode_length,
                "optimizer": "DNS",
            },
        )

    # -- jitted rollout returning reward + descriptor -------------------------
    @jax.jit
    def rollout_single(params, rng_key):
        """Roll out one episode. Returns (total_reward, ep_length, descriptor)."""
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
            return (env_st_next, obs_next, done_, new_hstate, rng_), (reward, done, obs_)

        _, (rewards, dones_seq, all_obs) = jax.lax.scan(
            _step, init_carry, None, length=episode_length
        )

        # Truncate rewards after first done
        any_done = jnp.any(dones_seq)
        first_done = jnp.argmax(dones_seq)
        first_done = jnp.where(any_done, first_done, episode_length)
        idxs = jnp.arange(episode_length)
        rewards = jnp.where(idxs > first_done, 0.0, rewards)

        total_reward = jnp.sum(rewards)

        # Behavioral descriptor: use the observation at the last valid step.
        # Extract the image global_info (small vector) from the final obs.
        # all_obs is the obs at each step BEFORE stepping (so index first_done
        # is the obs right when the episode ends).
        last_idx = jnp.clip(first_done, 0, episode_length - 1)
        # Use global_info from the PixelsObservation as the descriptor.
        # global_info is typically a 1D vector; we take the first 2 elements.
        if hasattr(all_obs, "global_info"):
            final_info = all_obs.global_info[last_idx]  # (info_dim,)
            # Pad to at least 2 dims, then take first 2
            padded = jnp.concatenate([final_info, jnp.zeros(2)])
            descriptor = padded[:2]
        else:
            # Fallback: flatten obs and take first 2 elements
            flat_obs = jax.tree_util.tree_leaves(all_obs)
            flat_concat = jnp.concatenate([jnp.ravel(x[last_idx]) for x in flat_obs])
            descriptor = flat_concat[:2]

        return total_reward, first_done, descriptor

    # -- batched evaluation ---------------------------------------------------
    @jax.jit
    def _eval_batch(flat_batch, rng_key):
        """Evaluate a batch of individuals over num_evals rollouts."""
        params_batch = reshaper.reshape(flat_batch)
        rep_keys = jr.split(rng_key, num_evals)

        def _eval_one_rep(rep_key):
            return jax.vmap(rollout_single, in_axes=(0, None))(params_batch, rep_key)

        # (num_evals, batch, ...) for each output
        all_fit, all_len, all_desc = jax.vmap(_eval_one_rep)(rep_keys)
        # -> (batch, num_evals, ...)
        return (
            jnp.transpose(all_fit),
            jnp.transpose(all_len),
            jnp.transpose(all_desc, (1, 0, 2)),
        )

    def eval_population(flat_pop, rng_key):
        """Evaluate population in chunks. Returns (fitnesses, ep_lengths, descriptors).

        Each is (pop, num_evals, ...) or (pop, num_evals) etc.
        """
        n = flat_pop.shape[0]
        fit_parts, len_parts, desc_parts = [], [], []
        for start in range(0, n, eval_batch_size):
            batch = flat_pop[start : start + eval_batch_size]
            rng_key, batch_key = jr.split(rng_key)
            f, l, d = _eval_batch(batch, batch_key)
            jax.block_until_ready(f)
            fit_parts.append(f)
            len_parts.append(l)
            desc_parts.append(d)
        return (
            jnp.concatenate(fit_parts, axis=0),
            jnp.concatenate(len_parts, axis=0),
            jnp.concatenate(desc_parts, axis=0),
        )

    # -- initialise population ------------------------------------------------
    rng, pop_rng = jr.split(rng)
    population = jr.normal(pop_rng, (pop_size, num_dims)) * 0.1

    # Initial evaluation
    print("Evaluating initial population...")
    rng, eval_key = jr.split(rng)
    all_fit, all_len, all_desc = eval_population(population, eval_key)

    # Average fitness across all rollouts for selection (like GA)
    fitnesses = jnp.mean(all_fit, axis=1)              # (pop,)
    descriptors = jnp.mean(all_desc, axis=1)           # (pop, desc_dim)
    novelties = compute_novelty(descriptors, k)
    mean_ep_len = float(jnp.mean(all_len))

    print(f"  Initial best (mean {num_evals} rollouts): {float(jnp.max(fitnesses)):.2f}")

    # -- training loop --------------------------------------------------------
    best_params = None
    start_time = time.time()

    print("\nStarting DNS training loop...")

    for gen in range(num_generations):
        # 1) Generate offspring via isoline variation
        rng, var_key = jr.split(rng)
        offspring = isoline_variation(population, var_key, iso_sigma, line_sigma, batch_size)

        # 2) Evaluate offspring
        rng, eval_key = jr.split(rng)
        off_all_fit, off_all_len, off_all_desc = eval_population(offspring, eval_key)
        off_fitnesses = jnp.mean(off_all_fit, axis=1)
        off_descriptors = jnp.mean(off_all_desc, axis=1)

        # 3) DNS selection (uses mean fitness across rollouts)
        population, fitnesses, descriptors, novelties = dns_selection(
            population, fitnesses, descriptors,
            offspring, off_fitnesses, off_descriptors,
            pop_size, k,
        )

        # Logging stats from selection fitnesses (no costly re-evaluation)
        fit_np = np.array(fitnesses)
        pop_host = np.array(population)

        gen_best = float(np.max(fit_np))
        gen_mean = float(np.mean(fit_np))
        best_idx = int(np.argmax(fit_np))
        mean_ep_len = 0.0  # ep lengths not tracked after selection

        # Always use the best from the current generation's population
        best_overall_fitness = gen_best
        best_params = pop_host[best_idx].copy()

        # Diversity metrics
        fitness_div = compute_fitness_diversity(fitnesses)
        genomic_div = compute_genomic_diversity(population)
        mean_novelty = float(jnp.nanmean(novelties))

        if use_wandb:
            wandb.log(
                {
                    "generation": gen,
                    "best_fitness": gen_best,
                    "mean_fitness": gen_mean,
                    "best_overall": best_overall_fitness,
                    "fitness_diversity": fitness_div,
                    "genomic_diversity": genomic_div,
                    "mean_novelty": mean_novelty,
                    "episode_length/mean": mean_ep_len,
                },
                step=gen,
            )

        elapsed = time.time() - start_time
        if gen % log_interval == 0 or gen == num_generations - 1:
            print(
                f"Gen {gen:4d}/{num_generations}  "
                f"best={gen_best:8.2f}  mean={gen_mean:8.2f}  "
                f"overall={best_overall_fitness:8.2f}  "
                f"nov={mean_novelty:.3f}  ep_len={mean_ep_len:6.0f}  "
                f"time={elapsed:6.1f}s"
            )

    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.1f}s")
    print(f"Best overall fitness (mean of {num_evals}): {best_overall_fitness:.2f}")

    # -- save checkpoint & GIFs -----------------------------------------------
    if output_dir and best_params is not None:
        best_params_tree = reshaper.reshape_single(jnp.array(best_params))
        ckpt_path = os.path.join(output_dir, f"dns_{bare_name}_best.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump(
                {
                    "params": jax.tree.map(np.array, best_params_tree),
                    "flat_params": best_params,
                    "best_fitness": float(best_overall_fitness),
                    "generation": num_generations,
                    "seed": seed,
                    "pop_size": pop_size,
                    "iso_sigma": iso_sigma,
                    "line_sigma": line_sigma,
                },
                f,
            )
        print(f"  Saved checkpoint: {ckpt_path}")

        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "best_fitness": float(best_overall_fitness),
                    "total_time_s": total_time,
                    "num_generations": num_generations,
                    "pop_size": pop_size,
                    "batch_size": batch_size,
                    "iso_sigma": iso_sigma,
                    "line_sigma": line_sigma,
                    "k": k,
                    "seed": seed,
                    "trial_idx": trial_idx,
                    "env_name": env_name,
                    "param_count": param_count,
                },
                f,
                indent=2,
            )
        print(f"  Saved metrics: {metrics_path}")

        # -- GIFs -------------------------------------------------------------
        print("\n  Generating 10 eval trajectory GIFs...")
        eval_env = make_kinetix_env(
            observation_type=config["observation_type"],
            action_type=config["action_type"],
            reset_fn=make_reset_fn_from_config(config, ep, static_ep),
            static_env_params=static_ep,
        )
        render_sep = eval_env.static_env_params.replace(downscale=4)
        pixel_renderer = jax.jit(make_render_pixels(ep, render_sep))

        best_params_tree = reshaper.reshape_single(jnp.array(best_params))

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

        gifs_dir = os.path.join(output_dir, "gifs")
        for eval_idx in range(10):
            print(f"    Eval trajectory {eval_idx}...", end="")
            eval_rng = jr.PRNGKey(seed * 100 + eval_idx)
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
                action = (
                    action.squeeze(0) if hasattr(action, "squeeze") else action[0]
                )

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
                gifs_dir, f"eval_{eval_idx:02d}_reward{int(total_reward)}.gif"
            )
            imageio.mimsave(gif_path, frames, fps=15, loop=0)
            print(f" reward={total_reward:.1f}, saved to {gif_path}")

        print(f"\n=== DNS project saving complete! ===")

    # -- cleanup --------------------------------------------------------------
    if use_wandb:
        wandb.finish()

    if tee_logger:
        sys.stdout = tee_logger.stdout
        tee_logger.close()

    return best_overall_fitness


# -- CLI ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DNS (Dominated Novelty Search) on Kinetix environments"
    )
    parser.add_argument(
        "--env", type=str, default=None,
        help="Environment name (e.g. h0_unicycle). If omitted, trains all 20.",
    )
    parser.add_argument("--gpu", type=str, default=None, help="GPU device ID")
    parser.add_argument("--pop_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_generations", type=int, default=200)
    parser.add_argument("--iso_sigma", type=float, default=0.05)
    parser.add_argument("--line_sigma", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=3, help="k nearest neighbors for novelty")
    parser.add_argument("--num_evals", type=int, default=3,
                        help="Rollouts averaged for reported fitness (default 3)")
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=128,
                        help="Chunk size for batched evaluation (reduce if OOM)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Kinetix-noncontinual-dns")
    parser.add_argument(
        "--project_dir", type=str, default=None,
        help="Root project dir for structured output (e.g. projects/kinetix)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # GPU already set before JAX import (top of file)

    envs = [args.env] if args.env else ENVIRONMENTS

    for env_name in envs:
        for trial in range(1, args.num_trials + 1):
            print(f"\n{'#'*60}")
            print(f"# Env: {env_name}  Trial: {trial}")
            print(f"{'#'*60}")
            train_dns(
                env_name,
                pop_size=args.pop_size,
                batch_size=args.batch_size,
                num_generations=args.num_generations,
                iso_sigma=args.iso_sigma,
                line_sigma=args.line_sigma,
                k=args.k,
                num_evals=args.num_evals,
                seed=args.seed + trial - 1,
                trial_idx=trial,
                project_dir=args.project_dir,
                use_wandb=not args.no_wandb,
                wandb_project=args.wandb_project,
                episode_length=args.episode_length,
                eval_batch_size=args.eval_batch_size,
                log_interval=args.log_interval,
            )


if __name__ == "__main__":
    main()
