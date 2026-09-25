"""Continual DNS (Dominated Novelty Search) training on Kinetix environments.

Built as a modified copy of ga_continual.py so that GA and DNS differ in
*exactly one thing*: the selection rule. Same network, same rollout, same
evaluation budget, same recorder, same task sequence. Anything else that
differed would show up in the GA-vs-DNS comparison as if it were the method.

Why it is not the previous implementation
-----------------------------------------
The old dns_continual.py (kept as dns_continual_qdax.py.unused) was built on
QDax: `from qdax.core.dns import DominatedNoveltySearch`, emitters, repertoires
and buffers, imported from a vendored `dependencies/` directory. That directory
does not exist in this repository and never has -- it is not in git, not in
.gitignore, and `import qdax` fails -- so that trainer could not run at all.

gymnax and mujoco never used QDax either: they use source/algorithms/ne/dns.py, a
self-contained port of the same algorithm ("exact port of QDax
dns_repertoire.py"). This file uses that shared module, so all three benchmarks
now run the same DNS code.

Selection is `dns_selection`: parents and offspring are pooled, dominated
novelty (mean descriptor distance to the k nearest *fitter* neighbours) is
computed on the pool, and the top `pop_size` survive. Offspring come from
`isoline_variation` (Iso+LineDD), not from the GA's crossover+mutation.

Trains sequentially through all 20 medium h-tasks WITHOUT resetting the
population between tasks. Uses the actor-only network (ActorOnlyPixelsRNN).

Usage:
    python experiments/dns_continual.py --gpu 0
    python experiments/dns_continual.py --gpu 1 --generations_per_task 100 --k 3
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
from source.studies.kinetix.ne_continual_io import ContinualRecorder, rerank_by_rollout

from kinetix.environment import make_reset_fn_from_config
from kinetix.environment.env import make_kinetix_env
from kinetix.models import ScannedRNN, make_network_from_config
from kinetix.render.renderer_pixels import make_render_pixels
from kinetix.util import normalise_config
from kinetix.util.saving import load_from_json_file

# ── DNS: the shared implementation, same one gymnax and brax use ─
# source/ lives at the repo root, which is not on the path when a trainer is
# run as a script rather than as a module, so put it there explicitly.
_REPO_FOR_COMMON = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _REPO_FOR_COMMON not in sys.path:
    sys.path.insert(0, _REPO_FOR_COMMON)
from source.algorithms.ne.dns import dns_selection, isoline_variation
from source.metrics.aurora import AuroraDescriptors, aurora_training_schedule
# Weight statistics from the SHARED module, same reason and same placement
# (after the sys.path line above). numpy path: DNS runs its generations in a
# Python loop, so the population is already concrete.
from source.metrics.weight_stats import population_weight_stats

from kinetix.util.behaviour import descriptor_from_rollout, step_features, subsample

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
        # The vendored Kinetix checkout, named outright. This used to be
        # `dirname(dirname(__file__))`, which worked only while our experiments
        # lived inside the vendored checkout. On 2026-09-08 they moved to
        # `source/studies/kinetix/` and the checkout moved to
        # `third_party/kinetix/`, so no relative walk connects them any more.
        os.path.join(REPO_ROOT, "third_party", "kinetix",
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

def train_dns_continual(
    stop_on_unsolved: int = 0,
    *,
    popsize: int = 1024,
    generations_per_task: int = 200,
    iso_sigma: float = 0.05,
    line_sigma: float = 0.5,
    k: int = 3,
    reeval_population: bool = False,
    descriptor: str = "aurora",
    handcrafted_kind: str = "duty_factor",
    traj_steps: int = 10,
    aurora_latent_dim: int = 6,
    aurora_lr: float = 1e-3,
    aurora_batch_size: int = 128,
    aurora_train_ratio: int = 8,
    seed: int = 0,
    trial_idx: int = 1,
    project_dir: str | None = None,
    use_wandb: bool = True,
    wandb_project: str = "Kinetix-continual-dns",
    episode_length: int = 1000,
    eval_reps: int = 3,
    evolve_reps: int = 3,
    eval_batch_size: int = 32,
    final_eval_reps: int = 20,
    behaviour_snapshots: int = 4,
    snapshot_pop: int = 32,
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

    # Actuator bindings are a property of env_size, identical on all 20
    # sub-tasks -- which is exactly why the duty-factor descriptor is
    # comparable across them.
    n_motor_bindings = int(static_ep0.num_motor_bindings)
    n_thruster_bindings = int(static_ep0.num_thruster_bindings)
    # `handcrafted_kind` is the expert descriptor; with --descriptor aurora it
    # is still computed and logged, but novelty is measured in the learned
    # latent space instead. Same arrangement as the gymnax DNS trainer, where
    # aurora is the default and handcrafted the fallback.
    handcrafted_dim = {
        "duty_factor": n_motor_bindings + n_thruster_bindings,
        "goal_trajectory": 5,
        "combined": n_motor_bindings + n_thruster_bindings + 5,
    }[handcrafted_kind]
    feature_dim = 7 + n_motor_bindings + n_thruster_bindings  # step_features width
    descriptor_dim = aurora_latent_dim if descriptor == "aurora" else handcrafted_dim
    print(f"Descriptor: {descriptor} ({descriptor_dim} dims); "
          f"handcrafted={handcrafted_kind} ({handcrafted_dim} dims); "
          f"{n_motor_bindings} motors + {n_thruster_bindings} thrusters")

    total_generations = generations_per_task * len(ENVIRONMENTS)

    # ── project dir setup ─────────────────────────────────────────
    output_dir = None
    tee_logger = None
    if project_dir:
        output_dir = os.path.join(
            project_dir, "continual", "dns", "all_tasks", f"trial_{trial_idx}"
        )
        os.makedirs(output_dir, exist_ok=True)
        gifs_dir = os.path.join(output_dir, "gifs")
        os.makedirs(gifs_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "train.log")
        tee_logger = Tee(log_file)
        sys.stdout = tee_logger

    print(f"\n=== Kinetix DNS Continual Training ===")
    print(f"  Trial: {trial_idx}")
    print(f"  Seed: {seed}")
    print(f"  Population size: {popsize}")
    print(f"  Generations per task: {generations_per_task}")
    print(f"  Total generations: {total_generations}")
    print(f"  iso_sigma: {iso_sigma}  line_sigma: {line_sigma}  k: {k}")
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
            name=f"DNS_continual_pop{popsize}_trial{trial_idx}_seed{seed}",
            config={
                "popsize": popsize,
                "generations_per_task": generations_per_task,
                "total_generations": total_generations,
                "iso_sigma": iso_sigma,
                "line_sigma": line_sigma,
                "reeval_population": bool(reeval_population),
                "k": k,
                "descriptor": descriptor,
                "seed": seed,
                "trial_idx": trial_idx,
                "param_count": param_count,
                "episode_length": episode_length,
                "eval_reps": eval_reps,
                "evolve_reps": evolve_reps,
                "optimizer": "DNS",
                "num_tasks": len(ENVIRONMENTS),
                "continual": True,
            },
        )

    # ── DNS population ────────────────────────────────────────────
    # A plain array, not an evosax state: DNS keeps a population and selects on
    # dominated novelty, so there is no strategy object to carry. Initialised
    # exactly as the GA's archive is, so the two start from the same
    # distribution and only the selection rule differs.
    print(f"  Selection: dominated novelty (k={k}), Iso+LineDD variation")
    print(f"  Descriptor: {descriptor}")
    rng, pop_rng = jr.split(rng)
    population = jr.uniform(pop_rng, (popsize, num_dims), minval=-1.0, maxval=1.0)
    # -inf marks "not yet evaluated"; the first generation fills these in.
    pop_fitness = jnp.full(popsize, -jnp.inf)
    pop_descriptors = jnp.zeros((popsize, descriptor_dim))
    # Behaviour trajectories travel with the survivors so AURORA can retrain on
    # them and every descriptor can be recomputed when the encoder changes.
    pop_observations = jnp.zeros((popsize, traj_steps, feature_dim))

    aurora, aurora_state, aurora_schedule = None, None, set()
    if descriptor == "aurora":
        aurora = AuroraDescriptors(
            obs_size=feature_dim, traj_steps=traj_steps,
            latent_dim=aurora_latent_dim, learning_rate=aurora_lr,
            batch_size=aurora_batch_size,
        )
        rng, aurora_key = jr.split(rng)
        aurora_state = aurora.init(aurora_key)
        aurora_schedule = aurora_training_schedule(
            total_generations, aurora_train_ratio)
        print(f"  AURORA: latent {aurora_latent_dim}, traj_steps {traj_steps}, "
              f"lr {aurora_lr}, retrain at {sorted(aurora_schedule)[:6]}...")

    # ── helper: build jitted rollout & eval for a given env ───────

    def make_rollout_and_eval(env, ep, init_env_state, n_reps=None):
        """Return a jitted eval_population fn for one environment.

        `n_reps` is how many rollouts each individual gets. The search only
        needs `evolve_reps` of them -- that is what drives selection -- but the
        end-of-task agent choice is an argmax over the whole population, and an
        argmax over noisy means reliably picks whichever individual got lucky.
        So the final selection is built with many more repeats while the search
        stays cheap. Paying the higher count every generation instead would
        cost 22h/trial at 10 repeats and 44h at 20, against 6.7h here.
        """
        _reps = total_reps if n_reps is None else n_reps

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
                # Behaviour features are read off the physics state BEFORE the
                # step, paired with the action taken from it.
                feat = step_features(env_st, action, n_motor_bindings)
                obs_next, env_st_next, reward, done, info = env.step(
                    step_rng, state=env_st, action=action, env_params=ep
                )
                done_ = jnp.expand_dims(done, axis=0)
                return ((env_st_next, obs_next, done_, new_hstate, rng_),
                        (reward, done, feat, action))

            _, (rewards, dones_seq, feats, actions) = jax.lax.scan(
                _step, init_carry, None, length=episode_length
            )
            any_done = jnp.any(dones_seq)
            first_done = jnp.argmax(dones_seq)
            first_done = jnp.where(any_done, first_done, episode_length)
            idxs = jnp.arange(episode_length)
            rewards = jnp.where(idxs > first_done, 0.0, rewards)
            # Steps at or before the first done are inside the episode.
            valid = (idxs <= first_done).astype(jnp.float32)
            hand_desc = descriptor_from_rollout(
                feats, actions, valid,
                n_motor_bindings, n_thruster_bindings, handcrafted_kind)
            # The sequence AURORA encodes: the low-dimensional physics/action
            # features, NOT the pixel observation. An LSTM auto-encoder over
            # 8192-dim frames would dominate the cost of the run and would
            # mostly encode rendering rather than behaviour.
            traj = subsample(feats, valid, traj_steps)
            return jnp.sum(rewards), first_done, hand_desc, traj

        def _eval_batch(flat_batch, rep_keys):
            params_batch = reshaper.reshape(flat_batch)

            def _eval_one_rep(rep_key):
                return jax.vmap(rollout_single, in_axes=(0, None))(params_batch, rep_key)

            all_fit, all_len, all_desc, all_traj = jax.vmap(_eval_one_rep)(rep_keys)
            # (reps, batch, ...) -> (batch, reps, ...)
            return (jnp.transpose(all_fit), jnp.transpose(all_len),
                    jnp.transpose(all_desc, (1, 0, 2)),
                    jnp.transpose(all_traj, (1, 0, 2, 3)))

        # Pre-compute number of batches (popsize must be divisible by eval_batch_size)
        num_batches = popsize // eval_batch_size
        assert popsize % eval_batch_size == 0, f"popsize ({popsize}) must be divisible by eval_batch_size ({eval_batch_size})"

        @jax.jit
        def eval_population(flat_pop, rng_key):
            """Evaluate population in chunks of eval_batch_size to avoid OOM.

            Fully JIT-compiled including scan, reshape, and batching.
            """
            rep_keys = jr.split(rng_key, _reps)

            # Reshape population to (num_batches, eval_batch_size, num_dims)
            batched_pop = flat_pop.reshape((num_batches, eval_batch_size, -1))

            def scan_fn(carry, batch):
                fit, length, desc, traj = _eval_batch(batch, rep_keys)
                return carry, (fit, length, desc, traj)

            _, (all_fits, all_lens, all_descs, all_trajs) = jax.lax.scan(
                scan_fn, None, batched_pop)
            # Descriptors are averaged over repeats, so a cell is the
            # individual's typical behaviour rather than one noisy rollout.
            # Trajectories keep only the first repeat: AURORA wants a sequence,
            # and averaging sequences across repeats would smear the behaviour
            # it is meant to encode.
            return (all_fits.reshape(popsize, -1),
                    all_lens.reshape(popsize, -1),
                    all_descs.reshape(popsize, _reps, -1).mean(axis=1),
                    all_trajs.reshape(popsize, _reps, traj_steps, -1)[:, 0])

        return eval_population, rollout_single

    # ── helper: generate GIFs for one task ────────────────────────

    def save_task_gifs(task_idx, env_name, config, env, ep, static_ep,
                       init_env_state, best_flat_params, num_gifs=10, eval_episode_length=None):
        # Use provided episode_length or fall back to env max_timesteps
        max_steps = eval_episode_length if eval_episode_length is not None else ep.max_timesteps
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

            for step in range(max_steps):
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
                task_gifs, f"rollout_{gif_idx:02d}_reward{float(total_reward):+.2f}.gif"
            )
            imageio.mimsave(gif_path, frames, fps=15, loop=0)
        print(f"    Saved {num_gifs} GIFs for task {task_idx} ({env_name}) → {task_gifs}")

    # ── training loop across tasks ────────────────────────────────
    print("Starting continual DNS training loop...")
    start_time = time.time()
    global_gen = 0
    best_fitness_ever = -jnp.inf
    best_params_ever = None

    # Records the run in the same layout as the gymnax continual trainers, so
    # forgetting / zero-shot / diversity are recoverable afterwards. Without it
    # the run leaves one global-best genome and no per-generation history, and
    # none of those can be reconstructed after the fact.
    recorder = ContinualRecorder(
        output_dir=output_dir,
        method="dns",
        envs=ENVIRONMENTS,
        generations_per_task=generations_per_task,
        seed=seed,
        trial=trial_idx,
        pop_size=popsize,
        behaviour_snapshots=behaviour_snapshots,
        snapshot_pop=snapshot_pop,
        descriptor=descriptor,
        config={
            "popsize": popsize,
            "iso_sigma": iso_sigma,
            "line_sigma": line_sigma,
            "reeval_population": bool(reeval_population),
            "k": k,
            "eval_reps": eval_reps,
            "evolve_reps": evolve_reps,
            "episode_length": episode_length,
            "param_count": int(param_count),
        },
    )

    for task_idx, env_name in enumerate(ENVIRONMENTS):
        config_t, env_t, init_es_t, static_ep_t, ep_t = envs_data[task_idx]
        eval_pop_fn, rollout_single_t = make_rollout_and_eval(env_t, ep_t, init_es_t, evolve_reps)
        eval_pop_fn_final, _ = make_rollout_and_eval(env_t, ep_t, init_es_t, final_eval_reps)

        # ── Re-evaluate the population on the new task at transitions ──
        # Fitness and descriptors are properties of the task, so carrying the
        # previous task's values into dns_selection would rank the population
        # on behaviour it no longer has.
        #
        # OFF BY DEFAULT SINCE 2026-09-08, and unconditional before it, so
        # every kinetix DNS run on disk was made with it on. Doing it means
        # knowing where the boundary is, and no other method in the comparison
        # is told that. The boundary-free cure is to re-score the repertoire
        # EVERY generation out of the same evaluation budget, which is
        # `DNSSearcher(refresh=True)` in source/studies/generalists/ne.py; it is not
        # implemented here.
        if task_idx > 0 and reeval_population:
            print(f"  Re-evaluating population ({popsize}) on new task...")
            rng, eval_key = jr.split(rng)
            _f, _l, _d, _o = eval_pop_fn(population, eval_key)
            pop_fitness = jnp.mean(_f[:, :evolve_reps], axis=1)
            pop_observations = _o
            # Re-encode with the current AURORA encoder rather than reusing the
            # handcrafted descriptors, or novelty would be measured in two
            # different spaces on either side of the switch.
            pop_descriptors = (aurora.encode(_o, aurora_state)
                               if aurora is not None else _d)
            jax.block_until_ready(pop_fitness)
            print(f"  Re-evaluated. Best fitness on new task: "
                  f"{float(jnp.max(pop_fitness)):.2f}")

        bare = env_name.split("/")[-1] if "/" in env_name else env_name

        print(f"\n{'='*60}")
        print(f"Task {task_idx}/{len(ENVIRONMENTS)-1}: {bare}  "
              f"(gens {global_gen}..{global_gen + generations_per_task - 1})")
        print(f"{'='*60}")
        print(f"  Compiling and running {generations_per_task} generations...", flush=True)

        # Define jitted training step for this task
        @jax.jit
        def train_step(carry, gen_idx, aurora_state_):
            (pop_, fit_, desc_, obs_), rng_, best_fit_ = carry
            rng_, ask_key, eval_key = jr.split(rng_, 3)

            # ask: Iso+LineDD offspring from the current population
            offspring = isoline_variation(
                pop_, ask_key, iso_sigma=iso_sigma, line_sigma=line_sigma,
                batch_size=popsize)
            all_fitnesses, all_ep_lengths, off_desc, off_obs = eval_pop_fn(
                offspring, eval_key)

            # The offspring descriptors must live in the SAME space as the
            # parents'. eval_pop_fn returns the HANDCRAFTED descriptor, but with
            # --descriptor aurora the parents carry AURORA latents, so passing
            # off_desc straight into selection compared two different spaces:
            # novelty distances between a parent and an offspring were
            # meaningless, and selection ranked on that. It never raised a shape
            # error because both happen to be 6-dimensional here -- duty_factor
            # is num_motor_bindings + num_thruster_bindings = 4 + 2, and the
            # AURORA latent is 6. gymnax encodes offspring and parents with the
            # same compute_descriptors(), which is what this restores.
            if aurora is not None:
                off_desc = aurora.encode(off_obs, aurora_state_)

            evolve_fit = jnp.mean(all_fitnesses[:, :evolve_reps], axis=1)
            report_fit = jnp.mean(all_fitnesses[:, :eval_reps], axis=1)
            mean_ep_lengths = jnp.mean(all_ep_lengths[:, :eval_reps], axis=1)

            best_report_idx = jnp.argmax(report_fit)
            best_report_fitness = report_fit[best_report_idx]
            best_fit_ = jnp.maximum(best_fit_, best_report_fitness)

            # tell: pool parents and offspring, keep the top popsize by
            # dominated novelty. Maximising here -- dns_selection takes
            # fitness as-is, unlike evosax's tell which minimises.
            # Fourth return is the dominated novelty of the survivors -- the
            # meta-fitness selection actually ranked on, worth logging.
            pop_, fit_, desc_, obs_, novelty_ = dns_selection(
                pop_, fit_, desc_, obs_,
                offspring, evolve_fit, off_desc, off_obs,
                population_size=popsize, k=k)

            # Metrics to collect
            best_rollouts = all_fitnesses[best_report_idx, :eval_reps]
            best_mean = jnp.mean(best_rollouts)
            pop_mean = jnp.mean(report_fit)
            ep_len_mean = jnp.mean(mean_ep_lengths)

            # Print every 10 generations using jax.debug.print
            jax.lax.cond(
                gen_idx % 10 == 0,
                lambda: jax.debug.print(
                    "Gen {g}  best_mean={bm:.2f}  pop_mean={pm:.2f}  ep_len={el:.0f}",
                    g=gen_idx, bm=best_mean, pm=pop_mean, el=ep_len_mean
                ),
                lambda: None
            )

            metrics = {
                "best_mean": best_mean,
                "best_min": jnp.min(best_rollouts),
                "best_max": jnp.max(best_rollouts),
                "pop_mean": pop_mean,
                "ep_len_mean": ep_len_mean,
                # Spread of the surviving population in descriptor space: the
                # quantity DNS selects on, so it is worth logging directly.
                "descriptor_spread": jnp.mean(jnp.std(desc_, axis=0)),
                # NaN for the fittest individuals (no fitter neighbour exists),
                # so use nanmean rather than reporting NaN for the generation.
                "dominated_novelty": jnp.nanmean(novelty_),
            }

            return ((pop_, fit_, desc_, obs_), rng_, best_fit_), metrics

        # Python generation loop, not a lax.scan over the whole sub-task.
        # AURORA retrains its encoder on a schedule and every stored descriptor
        # must then be re-encoded, which cannot happen inside a scan. This is
        # the same structure source/studies/gymnax/train_DNS_gymnax_continual.py uses.
        t0 = time.time()
        carry = ((population, pop_fitness, pop_descriptors, pop_observations),
                 rng, best_fitness_ever)

        for local_gen in range(generations_per_task):
            carry, gen_metrics = train_step(carry, local_gen, aurora_state)
            (population, pop_fitness, pop_descriptors, pop_observations), rng, _bf = carry

            if aurora is not None and (global_gen + local_gen) in aurora_schedule:
                rng, ae_key = jr.split(rng)
                aurora_state, ae_loss = aurora.train(
                    ae_key, pop_observations, aurora_state,
                    iteration=global_gen + local_gen)
                # Descriptors from the old encoder are not comparable with the
                # new one, so re-encode the whole population before the next
                # selection step.
                pop_descriptors = aurora.encode(pop_observations, aurora_state)
                carry = ((population, pop_fitness, pop_descriptors, pop_observations),
                         rng, _bf)
                print(f"    [aurora] retrained at gen {global_gen + local_gen}, "
                      f"loss={float(ae_loss):.5f}", flush=True)

            m = {k_: float(v_) for k_, v_ in gen_metrics.items()}
            # Weight statistics of the surviving population, alongside the
            # descriptor-space diversity DNS already reports.
            m.update(population_weight_stats(population))
            recorder.log_generation(task_idx, bare, local_gen, m)
            recorder.maybe_snapshot(task_idx, local_gen, population)
            if use_wandb:
                wandb.log({"generation": global_gen + local_gen,
                           "task": task_idx, "task_name": bare,
                           "best/mean": m["best_mean"], "best/min": m["best_min"],
                           "best/max": m["best_max"],
                           "population/mean": m["pop_mean"],
                           "episode_length/mean": m["ep_len_mean"],
                           "dns/descriptor_spread": m.get("descriptor_spread"),
                           "dns/dominated_novelty": m.get("dominated_novelty")},
                          step=global_gen + local_gen)
            if local_gen % 10 == 0 or local_gen == generations_per_task - 1:
                print(f"Gen {global_gen + local_gen:4d} (task {task_idx} "
                      f"local {local_gen:3d})  best={m['best_mean']:8.2f}  "
                      f"pop={m['pop_mean']:8.2f}  "
                      f"spread={m.get('descriptor_spread', float('nan')):.4f}",
                      flush=True)

        dt = time.time() - t0
        task_best_fitness = float(jnp.max(pop_fitness))

        # Score the surviving population once more so the reported number and
        # the saved agent come from the same evaluation.
        rng, eval_key = jr.split(rng)
        final_pop = population  # DNS keeps its population directly
        final_fitnesses, _fl, _fd, _fo = eval_pop_fn_final(final_pop, eval_key)
        final_mean_fit = jnp.mean(final_fitnesses[:, :final_eval_reps], axis=1)
        # Re-rank the top candidates with the DIRECT rollout -- the evaluator
        # that produces the GIFs and every reported number. See
        # ne_continual_io.rerank_by_rollout for why the batched score alone is
        # not safe to select on.
        rng, rr_key = jr.split(rng)
        task_best_params, task_best_fitness, _moved = rerank_by_rollout(
            final_pop, final_mean_fit, rollout_single_t, reshaper, rr_key,
            top_k=32, reps=final_eval_reps)
        if _moved:
            print(f"  [rerank] direct-rollout re-rank changed the saved agent "
                  f"(batched argmax was not the best under the reported evaluator)",
                  flush=True)

        # Update global best
        if task_best_fitness > best_fitness_ever:
            best_fitness_ever = task_best_fitness
            best_params_ever = task_best_params

        # Two agents per sub-task, as the gymnax tree records them:
        #   finalgen  -- best of this sub-task's final generation
        #   incumbent -- the elite-archive mean, what the search carries forward
        # They answer different questions, and `agent_sources` in results.json
        # records which is which.
        # DNS has no elite mean; the population mean is the analogous
        # 'what the search carries forward' agent.
        incumbent = jnp.mean(population, axis=0)
        recorder.end_task(
            task_idx, bare, task_best_params, incumbent,
            {
                "final_best_fitness": float(task_best_fitness),
                "final_pop_mean_fitness": float(jnp.mean(final_mean_fit)),
                "seconds": float(dt),
            },
        )
        # The population at the end of the sub-task is always on the pinned
        # snapshot schedule, so diversity at the switch is always captured.
        recorder.maybe_snapshot(task_idx, generations_per_task - 1, final_pop)

        global_gen += generations_per_task
        print(f"  Task {task_idx} ({bare}) finished in {dt:.1f}s.  task_best={float(task_best_fitness):.2f}")

        # ── end of task: save GIFs ────────────────────────────────
        # Verify the agent that is actually saved, independently of the
        # selection that chose it. task_best comes from an argmax over the
        # population's fitness; this rolls THAT agent out 16 times on its own.
        # A gap between the two is the difference between "the search found a
        # good policy" and "the policy we kept is good" -- DNS reported
        # task_best=1.28 on h3_car_thrust while every evaluation rollout
        # failed, and that gap is what this makes visible in the log.
        _vk = jr.split(rng, 17)[1:]
        # rollout_single takes a PYTREE, not the flat genome -- reshape first.
        _vtree = reshaper.reshape_single(task_best_params)
        _vr = jnp.array([rollout_single_t(_vtree, k)[0] for k in _vk])
        print(f"  [verify] saved agent over 16 rollouts: mean={float(jnp.mean(_vr)):+.3f} "
              f"min={float(jnp.min(_vr)):+.3f} max={float(jnp.max(_vr)):+.3f} "
              f"solved={int(jnp.sum(_vr >= 1.0))}/16  (training said {float(task_best_fitness):+.3f})",
              flush=True)

        # The same agent, scored by the BATCHED evaluator that chose it: tile it
        # into a full population and read any row. This splits the two ways the
        # numbers above can disagree --
        #   this says +1.13  -> eval_pop_fn_final is wrong, and since the same
        #                       evaluator produces evolve_fit, selection is too;
        #   this says -1.00  -> the evaluator is fine and the argmax/index that
        #                       picked task_best_params out of the population is.
        # On DNS seed 1 / h0_unicycle these disagreed by 2.1 reward with zero
        # variance on both sides, which no policy can do.
        rng, _tk = jr.split(rng)
        _tiled = jnp.broadcast_to(task_best_params[None, :], (popsize, task_best_params.shape[0]))
        _tf, _, _, _ = eval_pop_fn_final(_tiled, _tk)
        _tm = jnp.mean(_tf[:, :final_eval_reps], axis=1)
        print(f"  [verify-batched] same agent via eval_pop_fn_final: "
              f"row0={float(_tm[0]):+.3f} all_rows_equal={bool(jnp.all(_tm == _tm[0]))}",
              flush=True)

        save_task_gifs(
            task_idx, bare, config_t, env_t, ep_t, static_ep_t,
            init_es_t, task_best_params, eval_episode_length=episode_length,
        )


        # Fail fast, if asked -- same rule and same reasoning as GA's:
        # a configuration that cannot solve a level will never be the one that
        # solves all 20, so the remaining levels are wasted compute. Judged on
        # the SAVED agent's verify rollouts. BREAKS the loop rather than
        # exiting, so ContinualRecorder.finalise() still writes every output.
        _solved = int(jnp.sum(_vr >= 1.0))
        if stop_on_unsolved and _solved < stop_on_unsolved:
            print(f"\n  [stop_on_unsolved] level {task_idx} ({bare}) solved "
                  f"{_solved}/16 < {stop_on_unsolved}; abandoning this chain "
                  f"after {task_idx + 1} of {len(ENVIRONMENTS)} levels.\n"
                  f"  Outputs are still written -- this breaks the loop, it "
                  f"does not kill the run.", flush=True)
            break

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Continual training complete! Total time: {total_time:.1f}s")
    print(f"Best fitness ever: {float(best_fitness_ever):.2f}")
    print(f"{'='*60}")

    # ── save checkpoint ───────────────────────────────────────────
    if output_dir and best_params_ever is not None:
        best_tree = reshaper.reshape_single(best_params_ever)
        ckpt_path = os.path.join(output_dir, "dns_continual_best.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump(
                {
                    "params": jax.tree.map(np.array, best_tree),
                    "best_fitness": float(best_fitness_ever),
                    "total_generations": total_generations,
                    "generations_per_task": generations_per_task,
                    "seed": seed,
                    "popsize": popsize,
                    "iso_sigma": iso_sigma,
                    "line_sigma": line_sigma,
                    "reeval_population": bool(reeval_population),
                    "k": k,
                },
                f,
            )
        print(f"  Saved checkpoint: {ckpt_path}")

    # training_metrics.json is now the per-generation history (written by the
    # recorder), alongside results.json / checkpoints.npz / behaviour_snapshots.npz.
    # The run-level summary that used to occupy that filename moves into
    # results.json, where it sits beside the per-sub-task records.
    if output_dir:
        recorder.finalise(
            total_time,
            extra={
                "best_fitness_over_run": float(best_fitness_ever),
                "total_generations": total_generations,
            },
        )

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
        description="Continual DNS on Kinetix - sequential tasks, no population reset"
    )
    parser.add_argument("--gpu", type=str, default=None, help="GPU device ID")
    parser.add_argument("--popsize", type=int, default=1024)
    parser.add_argument("--generations_per_task", type=int, default=200)
    parser.add_argument("--behaviour_snapshots", type=int, default=4,
                        help="Populations saved per sub-task to behaviour_snapshots.npz "
                             "for the offline diversity analysis (0 disables). Pinned per "
                             "sub-task, so this is per sub-task, not per run.")
    parser.add_argument("--iso_sigma", type=float, default=0.05)
    parser.add_argument("--line_sigma", type=float, default=0.5)
    parser.add_argument("--reeval_population", type=int, default=0,
                        help="Re-score the population and re-encode its "
                             "descriptors on the new task at every boundary. "
                             "OFF by default since 2026-09-08: it needs to "
                             "know where the boundary is, and no other method "
                             "in the comparison is told. Was unconditional "
                             "before that date, so every kinetix DNS run on "
                             "disk was made with it on.")
    parser.add_argument("--k", type=int, default=3,
                        help="Neighbours used for dominated novelty")
    parser.add_argument("--descriptor", type=str, default="aurora",
                        choices=["aurora", "handcrafted"],
                        help="aurora: unsupervised descriptors learned online "
                             "(the paper default, and what gymnax/mujoco use). "
                             "handcrafted: the expert descriptor below.")
    parser.add_argument("--handcrafted_kind", type=str, default="duty_factor",
                        choices=["duty_factor", "goal_trajectory", "combined"],
                        help="Expert descriptor; used when --descriptor handcrafted, "
                             "and logged either way")
    parser.add_argument("--traj_steps", type=int, default=10)
    parser.add_argument("--aurora_latent_dim", type=int, default=6)
    parser.add_argument("--aurora_lr", type=float, default=1e-3)
    parser.add_argument("--aurora_batch_size", type=int, default=128)
    parser.add_argument("--aurora_train_ratio", type=int, default=8)
    parser.add_argument("--pop_size", type=int, dest="popsize",
                        help="Alias for --popsize, for the shared NE launcher")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=10)
    # Without this, `--num_trials 1 --seed N` always wrote trial_1, so two
    # concurrently launched seeds overwrote each other's output directory.
    parser.add_argument("--trial_idx", type=int, default=None,
                        help="Directory index to write under (trial_<N>). "
                             "Defaults to the position in the --num_trials "
                             "loop; set it when launching one seed per job.")
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument(
        "--eval_batch_size", type=int, default=32,
        help="Chunk size for batched population evaluation (default 128, reduce if OOM)",
    )
    parser.add_argument(
        "--final_eval_reps", type=int, default=20,
        help="Rollouts per individual in the END-OF-TASK selection. That step is "
             "an argmax over the whole population, so it needs a low-noise "
             "estimate or it picks the luckiest sample; the search itself "
             "keeps --evolve_reps.",
    )
    parser.add_argument(
        "--eval_reps", type=int, default=3,
        help="Number of rollouts averaged for reported fitness (default 3)",
    )
    parser.add_argument(
        "--evolve_reps", type=int, default=3,
        help="Number of rollouts averaged for evolution fitness / tell() (default 3)",
    )
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Kinetix-continual-dns")
    parser.add_argument(
        "--snapshot_pop", type=int, default=32,
        help="Population members kept per behaviour snapshot (0 = all). Full "
             "512-member snapshots are 2.3 GB each at kinetix network size.",
    )
    parser.add_argument(
        "--stop_on_unsolved", type=int, default=0,
        help="Abort the chain at the first level whose SAVED agent solves fewer "
             "than N of its 16 verify rollouts (0 = never abort). 9 is the "
             "majority rule. Breaks the loop, so all outputs are still written.",
    )
    parser.add_argument(
        "--num_tasks", type=int, default=None,
        help="Run only the first N levels of the 20-level chain. For a "
             "diversity estimate the whole chain is unnecessary -- but "
             "**stop the chain with this flag, never by killing the "
             "process**: ContinualRecorder.finalise() writes "
             "training_metrics.json, checkpoints.npz, behaviour_snapshots.npz "
             "and results.json at the end, so a kill loses every one of them. "
             "That is exactly how the 2026-08-03 drop lost its checkpoints and "
             "snapshots and left only train.log.",
    )
    parser.add_argument(
        "--project_dir", type=str, default=None,
        help="Root project dir for structured output (e.g. projects/kinetix)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Truncating the module-level list is what makes `--num_tasks` safe:
    # `len(ENVIRONMENTS)` is read in a dozen places (budget, headers,
    # results.json), so slicing here keeps every one of them consistent
    # instead of leaving the run claiming 20 levels while running 5.
    if args.num_tasks is not None:
        global ENVIRONMENTS
        if not 1 <= args.num_tasks <= len(ENVIRONMENTS):
            raise SystemExit(
                f"--num_tasks must be 1..{len(ENVIRONMENTS)}")
        ENVIRONMENTS = ENVIRONMENTS[:args.num_tasks]
        print(f"Running the first {len(ENVIRONMENTS)} levels: "
              f"{', '.join(ENVIRONMENTS)}")

    # GPU already set before JAX import (top of file)

    for trial in range(1, args.num_trials + 1):
        print(f"\n{'#'*60}")
        print(f"# Continual DNS  Trial: {trial}")
        print(f"{'#'*60}")
        train_dns_continual(
            stop_on_unsolved=args.stop_on_unsolved,
            popsize=args.popsize,
            generations_per_task=args.generations_per_task,
            behaviour_snapshots=args.behaviour_snapshots,
            snapshot_pop=args.snapshot_pop,
            iso_sigma=args.iso_sigma,
            line_sigma=args.line_sigma,
            k=args.k,
            reeval_population=bool(args.reeval_population),
            descriptor=args.descriptor,
            handcrafted_kind=args.handcrafted_kind,
            traj_steps=args.traj_steps,
            aurora_latent_dim=args.aurora_latent_dim,
            aurora_lr=args.aurora_lr,
            aurora_batch_size=args.aurora_batch_size,
            aurora_train_ratio=args.aurora_train_ratio,
            seed=args.seed + trial - 1,
            trial_idx=(trial if args.trial_idx is None else args.trial_idx),
            project_dir=args.project_dir,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            episode_length=args.episode_length,
            eval_reps=args.eval_reps,
            evolve_reps=args.evolve_reps,
            eval_batch_size=args.eval_batch_size,
            final_eval_reps=args.final_eval_reps,
        )


if __name__ == "__main__":
    main()
