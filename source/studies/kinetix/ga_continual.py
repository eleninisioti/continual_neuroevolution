"""Continual GA training on Kinetix environments.

Trains sequentially through all 20 medium h-tasks WITHOUT resetting the
population between tasks.  Each task receives ``--generations_per_task``
generations (default 200).  The ES state (elite archive) carries over.

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
import yaml

from flax.serialization import to_state_dict

import wandb

# ── Kinetix imports ──────────────────────────────────────────────
from source.studies.kinetix.ne_continual_io import ContinualRecorder, rerank_by_rollout

from kinetix.environment import make_reset_fn_from_config
from kinetix.environment.env import make_kinetix_env
from kinetix.models import ScannedRNN, make_network_from_config
from kinetix.util.behaviour import step_features, subsample
from kinetix.render.renderer_pixels import make_render_pixels
from kinetix.util import normalise_config
from kinetix.util.saving import load_from_json_file

# ── the one GA ───────────────────────────────────────────────────
# source/ lives at the repo root, which is not on the path when a trainer is
# run as a script rather than as a module, so put it there explicitly. This replaced
# experiments/simple_ga_elitist.py, a second copy of the same (mu+lambda)
# algorithm written against the evosax v2 base class.
_REPO_FOR_CORE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _REPO_FOR_CORE not in sys.path:
    sys.path.insert(0, _REPO_FOR_CORE)
from source.algorithms.ne.ga import SimpleGA
# Same reason, same placement: this must come AFTER the sys.path line above.
# Weight statistics come from the SHARED module rather than a local copy, so a
# kinetix column means exactly what the brax/gymnax/mujoco one means. The `_jax`
# variant because this population lives inside a lax.scan and never reaches the
# host -- identical formulas, NEAR_ZERO and keys to the numpy path.
from source.metrics.weight_stats import population_weight_stats_jax
# AURORA is the ONLY behavioural descriptor kinetix gets: the handcrafted,
# occupancy and action-frequency ones need environment semantics this suite has
# no definition for. It encodes the low-dimensional physics/action features per
# step, NOT the pixel frame -- an LSTM auto-encoder over 8192-dim frames would
# dominate the run and would mostly encode rendering rather than behaviour.
from source.metrics.aurora import AuroraDescriptors, aurora_training_schedule
from source.metrics.behaviour_descriptors import mean_pairwise_euclidean

# ── Constants ────────────────────────────────────────────────────

# Plasticity probes. 128 states matches what every other suite uses; the sample
# cap mirrors `population_dormancy`'s 64-of-512 in source/metrics.
PROBE_STATES = 128
DORMANCY_SAMPLE = 64
CHURN_SAMPLE = 64
# Pairwise quantities are O(n^2) in the sample, so this is smaller.
DIVERSITY_SAMPLE = 32
# Steps AURORA's LSTM auto-encoder sees per individual (DNS uses the same).
TRAJ_STEPS = 10

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

def train_ga_continual(
    *,
    popsize: int = 1024,
    generations_per_task: int = 200,
    sigma_init: float = 0.001,
    sigma_decay: float = 1.0,
    sigma_limit: float = 0.0001,
    crossover_rate: float = 0.2,
    reeval_archive: bool = False,
    stop_on_unsolved: int = 0,
    seed: int = 0,
    trial_idx: int = 1,
    project_dir: str | None = None,
    use_wandb: bool = True,
    wandb_project: str = "Kinetix-continual-ga",
    episode_length: int = 1000,
    eval_reps: int = 3,
    evolve_reps: int = 3,
    eval_batch_size: int = 32,
    final_eval_reps: int = 20,
    optimizer: str = "SimpleGA",
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
    # Turn on the actor network's dormant-unit counting. `count_dormant` is
    # derived from this inside make_network_from_config, and the actor-only
    # network publishes per-neuron scores with `sow` rather than changing its
    # return signature -- so this is an OBSERVER: logits and hidden state are
    # bit-identical with it on or off (verified), and nothing is fed back into
    # the search. Same criterion ("variability", std of activation) and tau as
    # the RL side, so an NE dormancy column means what an RL one means.
    for _c in envs_data:
        _c[0]["monitor_dormant"] = True
    config0["monitor_dormant"] = True
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

    # step_features' width: DNS derives it the same way.
    n_motor_bindings = int(static_ep0.num_motor_bindings)
    n_thruster_bindings = int(static_ep0.num_thruster_bindings)
    feature_dim = 7 + n_motor_bindings + n_thruster_bindings

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
    print(f"  Crossover rate: {crossover_rate}")
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
            name=f"GA_continual_pop{popsize}_trial{trial_idx}_seed{seed}",
            config={
                "popsize": popsize,
                "generations_per_task": generations_per_task,
                "total_generations": total_generations,
                "sigma_init": sigma_init,
                "sigma_decay": sigma_decay,
                "sigma_limit": sigma_limit,
                "crossover_rate": crossover_rate,
                # Whether the GA was told where the task boundaries are.
                # False for runs made after 2026-09-08; runs before it have no
                # entry and were made with the boundary re-score always on.
                "reeval_archive": bool(reeval_archive),
                "stop_on_unsolved": stop_on_unsolved,
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

    # ── strategy ──────────────────────────────────────────────────
    strategy = SimpleGA(
        popsize=popsize,
        num_dims=num_dims,
        sigma_init=sigma_init,
        sigma_decay=sigma_decay,
        sigma_limit=sigma_limit,
        cross_over_rate=crossover_rate,
    )
    es_params = strategy.default_params

    num_elites = strategy.num_elites
    print(f"  num_elites (archive size): {num_elites}")

    rng, init_rng = jr.split(rng)
    es_state = strategy.init(init_rng, es_params)

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
                obs_next, env_st_next, reward, done, info = env.step(
                    step_rng, state=env_st, action=action, env_params=ep
                )
                done_ = jnp.expand_dims(done, axis=0)
                feat = step_features(env_st, action, n_motor_bindings)
                return ((env_st_next, obs_next, done_, new_hstate, rng_),
                        (reward, done, feat))

            _, (rewards, dones_seq, feats) = jax.lax.scan(
                _step, init_carry, None, length=episode_length
            )
            any_done = jnp.any(dones_seq)
            first_done = jnp.argmax(dones_seq)
            first_done = jnp.where(any_done, first_done, episode_length)
            idxs = jnp.arange(episode_length)
            rewards = jnp.where(idxs > first_done, 0.0, rewards)
            valid = (idxs <= first_done).astype(jnp.float32)
            traj = subsample(feats, valid, TRAJ_STEPS)
            return jnp.sum(rewards), first_done, traj

        def _eval_batch(flat_batch, rep_keys):
            params_batch = reshaper.reshape(flat_batch)

            def _eval_one_rep(rep_key):
                return jax.vmap(rollout_single, in_axes=(0, None))(params_batch, rep_key)

            all_fit, all_len, all_traj = jax.vmap(_eval_one_rep)(rep_keys)
            # (reps, batch, ...) -> (batch, reps, ...); AURORA encodes the FIRST
            # rep's trajectory, matching what DNS feeds it.
            return (jnp.transpose(all_fit), jnp.transpose(all_len),
                    jnp.swapaxes(all_traj, 0, 1)[:, 0])

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
                fit, length, traj = _eval_batch(batch, rep_keys)
                return carry, (fit, length, traj)

            _, (all_fits, all_lens, all_trajs) = jax.lax.scan(
                scan_fn, None, batched_pop)
            # all_fits: (num_batches, eval_batch_size, total_reps)
            # Reshape back to (popsize, total_reps)
            return (all_fits.reshape(popsize, -1),
                    all_lens.reshape(popsize, -1),
                    all_trajs.reshape(popsize, TRAJ_STEPS, -1))

        @jax.jit
        def collect_probe_obs(params, rng_key, n_probe=PROBE_STATES):
            """A frozen batch of observations for the plasticity probes.

            Dormancy scores a unit by the VARIABILITY of its activation across a
            batch of states, so the batch has to be diverse: `env.reset` is
            overridden to a fixed initial state here, and a batch of identical
            observations would make every unit look dead. So the states come
            from an actual rollout.

            Observer discipline (CLAUDE.md): the key is split off a PRIVATE
            stream once per level and never returned to the search, so a run
            with these columns and one without are the same run.
            """
            rng_key, reset_key, step_key = jr.split(rng_key, 3)
            obs, env_state = env.reset(
                reset_key, env_params=ep, override_reset_state=init_env_state
            )
            dones = jnp.zeros(1, dtype=jnp.bool_)
            hstate = ScannedRNN.initialize_carry(1)

            def _step(carry, _):
                env_st, obs_, done_, hstate_, rng_ = carry
                rng_, act_rng, step_rng = jr.split(rng_, 3)
                ac_in = jax.tree.map(lambda x: x[None, None, ...], (obs_, done_))
                new_hstate, pi = network.apply(params, hstate_, ac_in)
                action = pi.sample(seed=act_rng)[0, 0, :]
                obs_next, env_st_next, reward, done, info = env.step(
                    step_rng, state=env_st, action=action, env_params=ep
                )
                done_ = jnp.expand_dims(done, axis=0)
                return (env_st_next, obs_next, done_, new_hstate, rng_), obs_

            _, obs_seq = jax.lax.scan(
                _step, (env_state, obs, dones, hstate, step_key), None,
                length=n_probe,
            )
            return obs_seq

        return eval_population, rollout_single, collect_probe_obs

    def population_dormancy(flat_sample, probe_obs, redo_tau):
        """Dormant-unit fractions over a sample of the population.

        Mirrors `source/metrics/plasticity.population_dormancy`: the mean
        over individuals of each one's dormant fraction, capped at a sample
        because the cost is linear in members. In-graph rather than host-side
        (the numpy version's approach) because this population lives inside a
        lax.scan and never reaches the host.

        The per-neuron scores come from the network's own `sow`, so the
        criterion and tau are the ones the RL side already uses -- an NE
        dormancy column and an RL one are the same measurement.
        """
        # `probe_obs` is a PixelsObservation pytree, not an array, so the batch
        # length comes off a leaf rather than the container.
        n_probe = jax.tree_util.tree_leaves(probe_obs)[0].shape[0]
        dones = jnp.zeros((n_probe, 1), dtype=jnp.bool_)
        hstate0 = ScannedRNN.initialize_carry(1)
        obs_in = jax.tree.map(lambda x: x[:, None, ...], probe_obs)

        def _one(flat):
            params = reshaper.reshape_single(flat)
            _, state = network.apply(
                params, hstate0, (obs_in, dones), mutable=["intermediates"],
            )
            leaves = jax.tree_util.tree_leaves(state["intermediates"])
            scores = jnp.concatenate([jnp.ravel(l) for l in leaves])
            return (jnp.mean(scores <= redo_tau).astype(jnp.float32),
                    jnp.mean(scores == 0.0).astype(jnp.float32))

        fr, zf = jax.vmap(_one)(flat_sample)
        return {
            "pop_dormant_fraction": jnp.mean(fr),
            "pop_zero_fraction": jnp.mean(zf),
        }

    def population_churn(parents, offspring, probe_obs):
        """Parent -> offspring action disagreement, the NE churn definition.

        CLAUDE.md is specific: NE churn pairs each offspring with the genome it
        was BRED FROM, not consecutive elites -- the elite changes lineage
        between generations, so elite-to-elite is not one network before and
        after one update. `SimpleGA.ask_with_parents` exists to supply that
        pairing.

        Discrete action space here, so the estimator is argmax disagreement,
        matching the gymnax NE cell of the churn table.
        """
        n_probe = jax.tree_util.tree_leaves(probe_obs)[0].shape[0]
        dones = jnp.zeros((n_probe, 1), dtype=jnp.bool_)
        hstate0 = ScannedRNN.initialize_carry(1)
        obs_in = jax.tree.map(lambda x: x[:, None, ...], probe_obs)

        def _greedy(flat):
            params = reshaper.reshape_single(flat)
            _, pi = network.apply(params, hstate0, (obs_in, dones))
            # MultiDiscreteActionDistribution does not implement `mode`, but it
            # holds one distrax.Categorical per action dimension, so the greedy
            # action is the argmax of each sub-distribution's logits. Stacked on
            # the last axis this is the same (..., n_dims) integer action the
            # rollout samples, which is what makes the disagreement fraction
            # comparable to the RL side's.
            return jnp.stack(
                [jnp.argmax(d.logits, axis=-1) for d in pi.distributions],
                axis=-1,
            )

        a = jax.vmap(_greedy)(parents)
        b = jax.vmap(_greedy)(offspring)
        return {"ne_pop_churn_action": jnp.mean((a != b).astype(jnp.float32))}

    def elite_churn(prev_elite, cur_elite, probe_obs):
        """Elite(t-1) -> elite(t) action disagreement.

        The SECONDARY column. CLAUDE.md keeps it precisely because it is NOT the
        primary: the elite changes lineage between generations, so this is not
        one network before and after one update the way parent -> offspring is.
        Reported so the two can be compared, not as a substitute.
        """
        n_probe = jax.tree_util.tree_leaves(probe_obs)[0].shape[0]
        dones = jnp.zeros((n_probe, 1), dtype=jnp.bool_)
        hstate0 = ScannedRNN.initialize_carry(1)
        obs_in = jax.tree.map(lambda x: x[:, None, ...], probe_obs)

        def _greedy(flat):
            params = reshaper.reshape_single(flat)
            _, pi = network.apply(params, hstate0, (obs_in, dones))
            return jnp.stack(
                [jnp.argmax(d.logits, axis=-1) for d in pi.distributions],
                axis=-1,
            )

        return {"ne_elite_churn_action":
                jnp.mean((_greedy(prev_elite) != _greedy(cur_elite)).astype(jnp.float32))}

    def population_diversity(flat_sample, fitnesses, ep_lengths, probe_obs):
        """The environment-AGNOSTIC half of the bd_* family.

        Handcrafted, occupancy and action-frequency descriptors are deliberately
        NOT here: they need a notion of "where the agent was" that is specific to
        each environment, and kinetix has no such definition. AURORA covers that
        role instead (per the call on this run). What remains needs only genomes,
        fitnesses and the frozen probe batch, so it carries over unchanged.

        Formulas follow source/metrics/behaviour_descriptors.py exactly --
        upper-triangular pairwise means, the same 1e-12 log guards -- so a
        kinetix bd_probe_js means what a gymnax one means.

        MultiDiscrete is the one departure: the shared code assumes a single
        (n, batch, A) logits array, while kinetix has one Categorical per action
        dimension. Each quantity is computed per dimension and averaged, which
        reduces to the shared definition when there is one dimension.
        """
        n = flat_sample.shape[0]
        n_probe = jax.tree_util.tree_leaves(probe_obs)[0].shape[0]
        dones = jnp.zeros((n_probe, 1), dtype=jnp.bool_)
        hstate0 = ScannedRNN.initialize_carry(1)
        obs_in = jax.tree.map(lambda x: x[:, None, ...], probe_obs)

        def _logits(flat):
            params = reshaper.reshape_single(flat)
            _, pi = network.apply(params, hstate0, (obs_in, dones))
            return [d.logits for d in pi.distributions]

        per_dim = jax.vmap(_logits)(flat_sample)
        mask = jnp.triu(jnp.ones((n, n)), k=1)
        denom = jnp.sum(mask)

        dis_acc, js_acc, ent_acc = 0.0, 0.0, 0.0
        for lg in per_dim:
            lp = jax.nn.log_softmax(lg, axis=-1)
            pr = jnp.exp(lp)
            act = jnp.argmax(lg, axis=-1)
            d = (act[:, None, ...] != act[None, ...]).astype(jnp.float32)
            dis_acc += jnp.sum(d.reshape(n, n, -1).mean(axis=-1) * mask) / denom
            p_i, p_j = pr[:, None], pr[None, :]
            m = 0.5 * (p_i + p_j)
            log_m = jnp.log(m + 1e-12)
            kl_i = jnp.sum(p_i * (jnp.log(p_i + 1e-12) - log_m), axis=-1)
            kl_j = jnp.sum(p_j * (jnp.log(p_j + 1e-12) - log_m), axis=-1)
            js = 0.5 * (kl_i + kl_j)
            js_acc += jnp.sum(js.reshape(n, n, -1).mean(axis=-1) * mask) / denom
            ent_acc += jnp.mean(-jnp.sum(pr * lp, axis=-1))
        k = float(len(per_dim))

        diff = flat_sample[:, None, :] - flat_sample[None, :, :]
        gd = jnp.sum(jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12) * mask) / denom

        return {
            "bd_probe_disagreement": dis_acc / k,
            "bd_probe_js": js_acc / k,
            "bd_probe_policy_entropy": ent_acc / k,
            "bd_genomic_diversity": gd,
            "bd_fitness_std": jnp.std(fitnesses),
            "bd_episode_steps_std": jnp.std(ep_lengths),
        }

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
    print("Starting continual GA training loop...")
    start_time = time.time()
    global_gen = 0
    # Diagnostics get their OWN key stream. CLAUDE.md requires the metrics to be
    # a pure observer -- if probes consumed `rng`, a run with these columns and
    # one without would diverge and the diagnostics would be changing the very
    # search they claim to describe.
    rng, probe_rng = jr.split(rng)
    # AURORA: an LSTM auto-encoder over the population's behaviour trajectories.
    # Its latent space is PER-RUN, so bd_aurora_diversity is comparable within a
    # run and across methods in the same run, never across runs -- the shared
    # module says so and the tables carry that caveat.
    total_generations_all = generations_per_task * len(ENVIRONMENTS)
    aurora = AuroraDescriptors(
        obs_size=feature_dim, traj_steps=TRAJ_STEPS, latent_dim=6,
        learning_rate=1e-3, batch_size=128,
    )
    rng, aurora_key = jr.split(rng)
    aurora_state = aurora.init(aurora_key)
    aurora_schedule = aurora_training_schedule(total_generations_all, 8)
    aurora_loss_last = None
    best_fitness_ever = -jnp.inf
    best_params_ever = None

    # Records the run in the same layout as the gymnax continual trainers, so
    # forgetting / zero-shot / diversity are recoverable afterwards. Without it
    # the run leaves one global-best genome and no per-generation history, and
    # none of those can be reconstructed after the fact.
    recorder = ContinualRecorder(
        output_dir=output_dir,
        method="ga",
        envs=ENVIRONMENTS,
        generations_per_task=generations_per_task,
        seed=seed,
        trial=trial_idx,
        pop_size=popsize,
        behaviour_snapshots=behaviour_snapshots,
        snapshot_pop=snapshot_pop,
        config={
            "popsize": popsize,
            "sigma_init": sigma_init,
            "eval_reps": eval_reps,
            "evolve_reps": evolve_reps,
            "episode_length": episode_length,
            "param_count": int(param_count),
        },
    )

    for task_idx, env_name in enumerate(ENVIRONMENTS):
        config_t, env_t, init_es_t, static_ep_t, ep_t = envs_data[task_idx]
        eval_pop_fn, rollout_single_t, probe_fn_t = make_rollout_and_eval(env_t, ep_t, init_es_t, evolve_reps)
        eval_pop_fn_final, _, _ = make_rollout_and_eval(env_t, ep_t, init_es_t, final_eval_reps)

        # ── frozen probe batch for this level's plasticity columns ───
        # Drawn ONCE per level, from a key split off `probe_rng`, a stream that
        # exists only for diagnostics and is never mixed back into `rng`. The
        # states come from rolling the archive's current best (or the initial
        # genome on level 0), so they are on-distribution for the policies being
        # scored rather than uniform noise.
        probe_rng, _pk = jr.split(probe_rng)
        _probe_src = (es_state.archive[0] if task_idx > 0
                      else reshaper.flatten_single(network_params))
        probe_obs_t = probe_fn_t(reshaper.reshape_single(_probe_src), _pk)
        dormant_tau = float(config0.get("redo_tau", 0.025))

        # ── Re-evaluate archive on the new task at task transitions ──
        # Boundary information, so gated and off by default. See the flag.
        if task_idx > 0 and reeval_archive:
            n_archive = es_state.archive.shape[0]
            print(f"  Re-evaluating archive ({n_archive} members) on new task...")
            # Pad archive to popsize so we can reuse the existing eval_pop_fn
            padded_pop = jnp.zeros((popsize, num_dims))
            padded_pop = padded_pop.at[:n_archive].set(es_state.archive)

            rng, eval_key = jr.split(rng)
            all_fit, _, _ = eval_pop_fn(padded_pop, eval_key)
            archive_fit = jnp.mean(all_fit[:n_archive, :evolve_reps], axis=1)

            # Negate because SimpleGA minimises (lower stored value = better)
            new_neg_fitness = -archive_fit

            # Re-sort archive by new fitness
            sorted_idx = jnp.argsort(new_neg_fitness)
            new_archive = es_state.archive[sorted_idx]
            new_fitness = new_neg_fitness[sorted_idx]

            # Reset best tracking for the new task
            best_archive_idx = jnp.argmin(new_neg_fitness)
            new_best_member = es_state.archive[best_archive_idx]
            new_best_fitness = new_neg_fitness[best_archive_idx]

            es_state = es_state.replace(
                archive=new_archive,
                fitness=new_fitness,
                mean=new_best_member,
                best_member=new_best_member,
                best_fitness=new_best_fitness,
            )
            jax.block_until_ready(es_state)
            print(f"  Archive re-evaluated. Best fitness on new task: {float(-new_best_fitness):.2f}")

        bare = env_name.split("/")[-1] if "/" in env_name else env_name

        print(f"\n{'='*60}")
        print(f"Task {task_idx}/{len(ENVIRONMENTS)-1}: {bare}  "
              f"(gens {global_gen}..{global_gen + generations_per_task - 1})")
        print(f"{'='*60}")
        print(f"  Compiling and running {generations_per_task} generations...", flush=True)

        # Define jitted training step for this task
        @jax.jit
        def train_step(carry, gen_idx):
            es_state_, rng_, best_fit_, prev_elite_ = carry
            rng_, ask_key, eval_key, tell_key = jr.split(rng_, 4)

            flat_pop, es_state_, parents_ = strategy.ask_with_parents(
                ask_key, es_state_, es_params)
            all_fitnesses, all_ep_lengths, all_trajs = eval_pop_fn(flat_pop, eval_key)

            evolve_fit = jnp.mean(all_fitnesses[:, :evolve_reps], axis=1)
            report_fit = jnp.mean(all_fitnesses[:, :eval_reps], axis=1)
            mean_ep_lengths = jnp.mean(all_ep_lengths[:, :eval_reps], axis=1)

            best_report_idx = jnp.argmax(report_fit)
            best_report_fitness = report_fit[best_report_idx]

            # Track best fitness (not params - too much memory)
            best_fit_ = jnp.maximum(best_fit_, best_report_fitness)

            # tell – SimpleGA minimises, negate evolve fitness
            es_state_ = strategy.tell(flat_pop, -evolve_fit, es_state_, es_params)

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
            }
            # Weight statistics of the population that was just evaluated --
            # the third plasticity signal beside dormancy and churn, and the one
            # that catches norms growing without bound over a long
            # non-stationary run while every unit is still nominally active.
            #
            # Measured on `flat_pop` (post-ask, pre-tell), which is the
            # generation the fitnesses above describe. Pure reductions on an
            # array already on device: no rollouts, no host transfer, nothing
            # fed back into the search.
            metrics.update(population_weight_stats_jax(flat_pop))
            # Dormancy on a sample of that same population. Pure forward passes
            # on a frozen probe batch; nothing is fed back into the search.
            metrics.update(population_dormancy(
                flat_pop[:DORMANCY_SAMPLE], probe_obs_t, dormant_tau))
            metrics.update(population_churn(
                parents_[:CHURN_SAMPLE], flat_pop[:CHURN_SAMPLE], probe_obs_t))
            metrics.update(elite_churn(
                prev_elite_, flat_pop[best_report_idx], probe_obs_t))
            metrics.update(population_diversity(
                flat_pop[:DIVERSITY_SAMPLE], report_fit, mean_ep_lengths,
                probe_obs_t))

            return ((es_state_, rng_, best_fit_, flat_pop[best_report_idx]),
                    (metrics, all_trajs))

        # Run generations in chunks to avoid OOM from storing all metrics
        # One generation per scan, not ten.
        #
        # train_step is redefined (and re-jitted) for every sub-task, so the
        # compile cost is paid 20 times per run. A 10-generation scan unrolls
        # 10 x popsize x eval_reps x 256 pixel-rendered steps into one graph,
        # which took >18 minutes to compile at popsize 128 and emitted nothing
        # in the meantime. DNS runs a single-generation step in a Python loop
        # and reached generation 160 in the same wall-clock time.
        #
        # The Python-loop overhead is negligible against a generation that
        # evaluates popsize x eval_reps full episodes, and it makes progress
        # visible every generation instead of every tenth.
        scan_chunk_size = 1
        num_chunks = generations_per_task // scan_chunk_size
        t0 = time.time()
        # The 4th slot is last generation's elite, for ne_elite_churn_action.
        # Seeded with the archive's best so generation 0 compares against the
        # population it actually came from rather than a zero vector.
        carry = (es_state, rng, best_fitness_ever, es_state.archive[0])
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * scan_chunk_size
            carry, (chunk_metrics, chunk_trajs) = jax.lax.scan(
                train_step, carry, jnp.arange(chunk_start, chunk_start + scan_chunk_size)
            )
            jax.block_until_ready(chunk_metrics)

            # Record every generation to disk, not just to wandb. This loop used
            # to run only `if use_wandb`, so a --no_wandb run kept no history at
            # all and a wandb run kept it only server-side.
            for local_idx in range(scan_chunk_size):
                gen_in_task = chunk_start + local_idx
                gen_num = global_gen + gen_in_task
                metrics_i = jax.tree.map(lambda x: float(x[local_idx]), chunk_metrics)
                # Everything train_step returned, not a hand-listed subset:
                # the weight-statistic keys are defined in the shared module and
                # listing them again here is how a column silently stops being
                # written when one is added.
                _rec = dict(metrics_i)
                # AURORA is host-side: encoding and auto-encoder training are
                # numpy/optax work that cannot live inside the scan.
                _traj = chunk_trajs[local_idx]
                _gen_global = global_gen + gen_in_task
                if _gen_global in aurora_schedule:
                    rng, _ak = jr.split(rng)
                    aurora_state, aurora_loss_last = aurora.train(
                        _ak, _traj, aurora_state, iteration=_gen_global)
                _desc = aurora.encode(_traj, aurora_state)
                _rec["bd_aurora_diversity"] = mean_pairwise_euclidean(
                    jax.device_get(_desc), max_n=DIVERSITY_SAMPLE, seed=seed)
                if aurora_loss_last is not None:
                    _rec["bd_aurora_loss"] = float(aurora_loss_last)
                # `ne_churn` is the headline column and `ne_churn_kind` names the
                # estimator, so a reader can tell which of the four definitions
                # in CLAUDE.md produced it. Added here rather than in train_step
                # because a string cannot live in a jitted metrics pytree.
                _rec["ne_churn"] = _rec.get("ne_pop_churn_action")
                _rec["ne_churn_kind"] = "action_disagreement_parent_offspring"
                _rec["pop_dormant_sample"] = min(DORMANCY_SAMPLE, popsize)
                recorder.log_generation(task_idx, bare, gen_in_task, _rec)
                if use_wandb:
                    wandb.log(
                        {
                            "generation": gen_num,
                            "task": task_idx,
                            "task_name": bare,
                            "best/mean": metrics_i["best_mean"],
                            "best/min": metrics_i["best_min"],
                            "best/max": metrics_i["best_max"],
                            "population/mean": metrics_i["pop_mean"],
                            "episode_length/mean": metrics_i["ep_len_mean"],
                        },
                        step=gen_num,
                    )
        
        es_state, rng, task_best_fitness, _last_elite = carry
        dt = time.time() - t0

        # Get best params from final population (re-evaluate to find best)
        rng, eval_key = jr.split(rng)
        final_pop, _ = strategy.ask(rng, es_state, es_params)  # Get current population
        # High-repeat evaluation: this argmax decides which agent is saved and
        # reported, so its fitness estimate has to be low-noise.
        final_fitnesses, _, _ = eval_pop_fn_final(final_pop, eval_key)
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
            print("  [rerank] direct-rollout re-rank changed the saved agent",
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
        incumbent = strategy.elite_mean(es_state)
        # Snapshot before end_task: end_task flushes everything to disk, so a
        # snapshot taken after it would not appear until the *next* sub-task's
        # flush -- and would be lost entirely if the run stopped in between.
        recorder.maybe_snapshot(task_idx, generations_per_task - 1, final_pop)
        recorder.end_task(
            task_idx, bare, task_best_params, incumbent,
            {
                "final_best_fitness": float(task_best_fitness),
                "final_pop_mean_fitness": float(jnp.mean(final_mean_fit)),
                "seconds": float(dt),
            },
        )

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

        save_task_gifs(
            task_idx, bare, config_t, env_t, ep_t, static_ep_t,
            init_es_t, task_best_params, eval_episode_length=episode_length,
        )

        # Fail fast, if asked. Judged on the SAVED agent's verify rollouts, not
        # on task_best -- the two can disagree, which is the whole reason the
        # verify pass above exists.
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
        description="Continual GA on Kinetix - sequential tasks, no population reset"
    )
    parser.add_argument("--gpu", type=str, default=None, help="GPU device ID")
    parser.add_argument("--popsize", type=int, default=1024)
    parser.add_argument("--generations_per_task", type=int, default=200)
    parser.add_argument("--behaviour_snapshots", type=int, default=4,
                        help="Populations saved per sub-task to behaviour_snapshots.npz "
                             "for the offline diversity analysis (0 disables). Pinned per "
                             "sub-task, so this is per sub-task, not per run.")
    parser.add_argument("--sigma_init", type=float, default=0.001)
    # SimpleGA multiplies sigma by this every generation, floored at
    # --sigma_limit. 1.0 (the default) means NO annealing ever: the step that
    # gets the search out of the initial region is the same step it must later
    # refine with. On kinetix h0 a large sigma is what lifts training from
    # -1.00 to -0.13, so the inability to shrink it afterwards is the obvious
    # next suspect once the large-sigma probes plateau.
    parser.add_argument("--sigma_decay", type=float, default=1.0)
    parser.add_argument("--sigma_limit", type=float, default=0.0001)
    parser.add_argument("--crossover_rate", type=float, default=0.2)
    parser.add_argument("--reeval_archive", type=int, default=0,
                        help="Re-score the elite archive on the new task at "
                             "every boundary and re-sort it. OFF by default "
                             "since 2026-09-08: doing it means knowing where "
                             "the boundary is, and no other method in the "
                             "comparison is told. Was unconditional before "
                             "that date, so every kinetix GA run on disk was "
                             "made with it on. The boundary-free cure for the "
                             "stale archive is to re-score every generation "
                             "out of the same budget "
                             "(--refresh_archive in "
                             "source/studies/gymnax/train_GA_gymnax_continual.py); "
                             "that is not implemented here.")
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
    parser.add_argument(
        "--optimizer", type=str, default="SimpleGA", choices=["SimpleGA"],
        help="Kept so old command lines still parse; there is one GA now "
             "(source/algorithms/ne/ga.py).",
    )
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Kinetix-continual-ga")
    parser.add_argument(
        "--snapshot_pop", type=int, default=32,
        help="Population members kept per behaviour snapshot (0 = all). Full "
             "512-member snapshots are 2.3 GB each at kinetix network size.",
    )
    parser.add_argument(
        "--stop_on_unsolved", type=int, default=0,
        help="Abort the chain at the first level whose SAVED agent solves fewer "
             "than N of its 16 verify rollouts (0 = never abort, the default). "
             "9 is the majority rule check_ne_solved.py uses.\n"
             "This exists for hyper-parameter search: a configuration that "
             "cannot solve level 0 will never be the configuration that solves "
             "all 20, so the remaining 19 levels are ~28 h of wasted compute "
             "per chain. Aborting BREAKS the task loop rather than exiting, so "
             "ContinualRecorder.finalise() still runs and every output is "
             "written -- unlike killing the process, which leaves only "
             "train.log.",
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
        print(f"# Continual GA  Trial: {trial}")
        print(f"{'#'*60}")
        train_ga_continual(
            popsize=args.popsize,
            generations_per_task=args.generations_per_task,
            behaviour_snapshots=args.behaviour_snapshots,
            snapshot_pop=args.snapshot_pop,
            sigma_init=args.sigma_init,
            sigma_decay=args.sigma_decay,
            sigma_limit=args.sigma_limit,
            crossover_rate=args.crossover_rate,
            stop_on_unsolved=args.stop_on_unsolved,
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
            optimizer=args.optimizer,
            reeval_archive=bool(args.reeval_archive),
        )


if __name__ == "__main__":
    main()
