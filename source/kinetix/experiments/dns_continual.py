"""Continual DNS (Dominated Novelty Search) training on Kinetix environments.

Trains sequentially through all 20 medium h-tasks WITHOUT resetting the
population between tasks.  Each task receives ``--generations_per_task``
generations (default 200).  The population carries over.

Uses QDax's DominatedNoveltySearch with the Kinetix actor-only network
(ActorOnlyPixelsRNN).  Isoline variation via QDax's MixingEmitter,
scan-based training loop.

Usage:
    python experiments/dns_continual.py --gpu 0
    python experiments/dns_continual.py --gpu 1 --generations_per_task 100
"""

import argparse
import functools
import gc
import json
import os
import pickle
import sys
import time
from typing import List, NamedTuple, Optional, Tuple

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
from PIL import Image

import wandb

# -- Add dependencies/ to path so we pick up the local QDax (with DNS) --------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
_DEPS_DIR = os.path.join(_REPO_ROOT, "dependencies")
if _DEPS_DIR not in sys.path:
    sys.path.insert(0, _DEPS_DIR)

# -- Kinetix imports ----------------------------------------------------------
from kinetix.environment import make_reset_fn_from_config
from kinetix.environment.env import make_kinetix_env
from kinetix.models import ScannedRNN, make_network_from_config
from kinetix.render.renderer_pixels import make_render_pixels, PixelsObservation
from kinetix.util import normalise_config
from kinetix.util.saving import load_from_json_file

# -- QDax imports -------------------------------------------------------------
from qdax.core.dns import DominatedNoveltySearch
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import CSVLogger, default_qd_metrics

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


# -- PolicyState (genotype for neuroevolution) --------------------------------

class PolicyState(NamedTuple):
    """Wraps network parameters + RNN hidden state."""
    network_params: dict
    rnn_state: Optional[jax.Array]
    n_dormant: Optional[jax.Array] = jnp.array([0.0])


def create_kinetix_policy_state(network, key, env, env_params, network_params=None):
    """Create initial network_params (used to build PolicyState)."""
    if network_params is None:
        obsv, _ = jax.vmap(env.reset, (0, None))(jr.split(key, 1), env_params)
        dones = jnp.zeros(1, dtype=jnp.bool_)
        init_hstate = ScannedRNN.initialize_carry(1)
        init_x = jax.tree.map(lambda x: x[None, ...], (obsv, dones))
        network_params = network.init(key, init_hstate, init_x)
    return network_params


def apply_kinetix_policy(obs, state, done, key, network):
    """Forward pass through the Kinetix network.  Returns (action, updated_state)."""
    # Extract done scalar
    if done.shape == ():
        done_scalar = done
    else:
        done_scalar = done[0] if done.shape[0] == 1 else done

    # Add batch dimension to PixelsObservation fields if needed
    if hasattr(obs, "image") and hasattr(obs, "global_info"):
        image = obs.image
        global_info = obs.global_info
        if len(image.shape) == 3:          # (H, W, C) -> (1, H, W, C)
            image = image[None, ...]
        if len(global_info.shape) == 1:    # (d,)  -> (1, d)
            global_info = global_info[None, :]
        obs = PixelsObservation(image=image, global_info=global_info)

    # Prepare input: add time dimension  (time=1, batch=1, ...)
    ac_in = jax.tree.map(lambda x: x[None, ...], (obs, done_scalar))
    rnn_state = state.rnn_state

    # Forward pass
    hstate, pi = network.apply(state.network_params, rnn_state, ac_in)
    action = pi.sample(seed=key)

    # Remove time and batch dims -> (action_dim,)
    action = jax.tree.map(
        lambda x: x[0, 0] if len(x.shape) >= 2 else x[0],
        action,
    )

    new_rnn_state = hstate[0] if isinstance(hstate, tuple) else hstate
    updated_state = state._replace(rnn_state=new_rnn_state)
    return action, updated_state


# -- Observation helpers ------------------------------------------------------

def flatten_observation(obs):
    """Flatten any observation type into a single 1-D vector."""
    if isinstance(obs, dict):
        parts = []
        for v in obs.values():
            if hasattr(v, "shape"):
                parts.append(jnp.ravel(v))
            else:
                parts.append(jnp.array([v]))
        return jnp.concatenate(parts)

    if hasattr(obs, "__dict__"):
        try:
            parts = []
            for v in obs.__dict__.values():
                if hasattr(v, "shape"):
                    parts.append(jnp.ravel(v))
                elif isinstance(v, (int, float)):
                    parts.append(jnp.array([v]))
            if parts:
                return jnp.concatenate(parts)
        except Exception:
            pass

    if hasattr(obs, "_fields") or hasattr(obs, "__dataclass_fields__"):
        parts = []
        fields = getattr(obs, "_fields", None) or getattr(
            obs, "__dataclass_fields__", {}
        )
        field_names = fields.keys() if isinstance(fields, dict) else (fields or [])
        for fn in field_names:
            try:
                v = getattr(obs, fn)
                if hasattr(v, "shape"):
                    parts.append(jnp.ravel(v))
                elif isinstance(v, (int, float)):
                    parts.append(jnp.array([v]))
            except Exception:
                pass
        if parts:
            return jnp.concatenate(parts)

    if hasattr(obs, "shape"):
        return obs.reshape(-1) if len(obs.shape) > 1 else obs
    return jnp.array(obs).reshape(-1)


# -- Environment step / reset closures ---------------------------------------

def create_play_step_fn(env, env_params, network):
    """Return a play_step_fn closure for one Kinetix environment."""

    def play_step_fn(obs, env_state, policy_state, done, key):
        key, subkey = jax.random.split(key)
        action, updated_policy_state = apply_kinetix_policy(
            obs, policy_state, done, subkey, network
        )

        key, subkey = jax.random.split(key)
        next_obs, next_env_state, reward, step_done, info = env.step(
            subkey, env_state, action, env_params
        )

        # Descriptor: first 2 elements of flattened obs
        flat_obs = flatten_observation(obs)
        descriptor = (
            flat_obs[:2]
            if flat_obs.shape[0] >= 2
            else jnp.pad(flat_obs, (0, 2 - flat_obs.shape[0]))[:2]
        )
        next_flat_obs = flatten_observation(next_obs)
        next_descriptor = (
            next_flat_obs[:2]
            if next_flat_obs.shape[0] >= 2
            else jnp.pad(next_flat_obs, (0, 2 - next_flat_obs.shape[0]))[:2]
        )

        transition = QDTransition(
            obs=obs,
            next_obs=next_obs,
            rewards=reward,
            dones=step_done,
            actions=action,
            truncations=jnp.array(False),
            state_desc=descriptor,
            next_state_desc=next_descriptor,
        )
        return next_obs, next_env_state, updated_policy_state, step_done, key, transition

    return play_step_fn


def create_env_reset_fn(env, env_params, init_env_state):
    """Return an env_reset_fn closure for one Kinetix environment."""

    def env_reset_fn(rng):
        rng, key_reset = jax.random.split(rng)
        obs, state = env.reset(
            key_reset, env_params=env_params, override_reset_state=init_env_state
        )
        return obs, state

    return env_reset_fn


# -- Episode rollout ----------------------------------------------------------

def generate_kinetix_unroll(
    init_obs, init_env_state, policy_state, key, episode_length, play_step_fn
):
    """Run one episode.  RNN state is reset at the start."""
    # Always reset RNN hidden state at episode start
    init_rnn_state = ScannedRNN.initialize_carry(1)
    if isinstance(init_rnn_state, tuple):
        init_rnn_state = init_rnn_state[0]
    policy_state = policy_state._replace(rnn_state=init_rnn_state)

    def step_fn(carry, _):
        obs, env_state, policy_state, done, key = carry
        next_obs, next_env_state, policy_state, done, key, transition = play_step_fn(
            obs, env_state, policy_state, done, key
        )
        return (next_obs, next_env_state, policy_state, done, key), transition

    done = jnp.array(False)
    (final_obs, final_state, final_ps, _, _), transitions = jax.lax.scan(
        step_fn,
        (init_obs, init_env_state, policy_state, done, key),
        (),
        length=episode_length,
    )
    return (final_obs, final_state), transitions, final_ps


# -- Scoring functions --------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("episode_length", "play_reset_fn", "play_step_fn", "descriptor_extractor", "num_evals"),
)
def scoring_function_kinetix(
    policies_params,
    key,
    episode_length,
    play_reset_fn,
    play_step_fn,
    descriptor_extractor,
    num_evals=1,
):
    """Evaluate a batch of policies, averaging over num_evals rollouts."""

    def _single_eval(policies_params, key):
        """Run one rollout per individual and return fitnesses + descriptors."""
        key, subkey = jax.random.split(key)
        if hasattr(policies_params, "rnn_state"):
            rnn = policies_params.rnn_state
            bs = rnn[0].shape[0] if isinstance(rnn, tuple) else rnn.shape[0]
        else:
            bs = jax.tree.leaves(policies_params.network_params)[0].shape[0]

        keys = jax.random.split(subkey, bs)
        init_obs, init_states = jax.vmap(play_reset_fn)(keys)

        unroll_fn = functools.partial(
            generate_kinetix_unroll,
            episode_length=episode_length,
            play_step_fn=play_step_fn,
        )
        keys = jax.random.split(key, bs)
        (final_obs, final_states), transitions, _ = jax.vmap(
            lambda obs, state, ps, k: unroll_fn(obs, state, ps, k)
        )(init_obs, init_states, policies_params, keys)

        is_done = jnp.clip(jnp.cumsum(transitions.dones, axis=1), 0, 1)
        mask = jnp.roll(is_done, 1, axis=1).at[:, 0].set(0)
        fitnesses = jnp.sum(transitions.rewards * (1.0 - mask), axis=1)
        descriptors = descriptor_extractor(transitions, mask)
        return fitnesses, descriptors

    # Run num_evals rollouts with different keys and average
    eval_keys = jax.random.split(key, num_evals)

    def _eval_one_rep(rep_key):
        return _single_eval(policies_params, rep_key)

    all_fitnesses, all_descriptors = jax.vmap(_eval_one_rep)(eval_keys)
    # all_fitnesses: (num_evals, batch_size), all_descriptors: (num_evals, batch_size, desc_dim)

    fitnesses = jnp.mean(all_fitnesses, axis=0)
    descriptors = jnp.mean(all_descriptors, axis=0)

    return fitnesses, descriptors, {}


@functools.partial(
    jax.jit,
    static_argnames=("episode_length", "play_reset_fn", "play_step_fn", "descriptor_extractor", "num_evals"),
)
def scoring_function_kinetix_lightweight(
    policies_params,
    key,
    episode_length,
    play_reset_fn,
    play_step_fn,
    descriptor_extractor,
    num_evals=1,
):
    """Lightweight scoring (no transitions) – used for re-evaluation.
    Averages fitness and descriptors over num_evals rollouts."""

    def _single_eval(policies_params, key):
        key, subkey = jax.random.split(key)
        if hasattr(policies_params, "rnn_state"):
            rnn = policies_params.rnn_state
            bs = rnn[0].shape[0] if isinstance(rnn, tuple) else rnn.shape[0]
        else:
            bs = jax.tree.leaves(policies_params.network_params)[0].shape[0]

        keys = jax.random.split(subkey, bs)
        init_obs, init_states = jax.vmap(play_reset_fn)(keys)

        unroll_fn = functools.partial(
            generate_kinetix_unroll,
            episode_length=episode_length,
            play_step_fn=play_step_fn,
        )
        keys = jax.random.split(key, bs)
        (final_obs, final_states), transitions, _ = jax.vmap(
            lambda obs, state, ps, k: unroll_fn(obs, state, ps, k)
        )(init_obs, init_states, policies_params, keys)

        is_done = jnp.clip(jnp.cumsum(transitions.dones, axis=1), 0, 1)
        mask = jnp.roll(is_done, 1, axis=1).at[:, 0].set(0)
        fitnesses = jnp.sum(transitions.rewards * (1.0 - mask), axis=1)
        descriptors = descriptor_extractor(transitions, mask)
        return fitnesses, descriptors

    eval_keys = jax.random.split(key, num_evals)

    def _eval_one_rep(rep_key):
        return _single_eval(policies_params, rep_key)

    all_fitnesses, all_descriptors = jax.vmap(_eval_one_rep)(eval_keys)

    fitnesses = jnp.mean(all_fitnesses, axis=0)
    descriptors = jnp.mean(all_descriptors, axis=0)

    return fitnesses, descriptors, {}


def descriptor_extractor_fn(transitions, mask):
    """Extract descriptor at the last valid timestep."""
    valid_indices = jnp.argmax(mask, axis=1) - 1
    valid_indices = jnp.clip(valid_indices, 0, transitions.next_state_desc.shape[1] - 1)
    batch_size = transitions.next_state_desc.shape[0]
    return transitions.next_state_desc[jnp.arange(batch_size), valid_indices]


# -- Config loading -----------------------------------------------------------

def load_base_config():
    yaml_candidates = [
        os.path.join(REPO_ROOT, "ga_example", "kinetix_config_pixels.yaml"),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "kinetix_config_pixels.yaml",
        ),
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


# -- GIF helper ---------------------------------------------------------------

def create_gif_from_frames(frames: List[jnp.ndarray], outfile: str, fps: int = 10):
    """Create a GIF from a list of frames."""
    if not frames:
        return
    pil_frames = [Image.fromarray(np.array(f)) for f in frames]
    dirname = os.path.dirname(outfile)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    pil_frames[0].save(
        outfile,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"Saved GIF to: {outfile}")


# -- Re-evaluation on task switch --------------------------------------------

def reevaluate_repertoire(
    repertoire, scoring_fn, key, mixing_emitter, metrics_function, population_size, k,
    eval_batch_size=128,
):
    """
    Re-evaluate all individuals in the repertoire with a new scoring function.

    Uses jax.lax.scan for sequential batched evaluation to manage memory.
    Invalid individuals (fitness == -inf) are kept as invalid.
    """
    genotypes = repertoire.genotypes
    valid_mask = repertoire.fitnesses[:, 0] != -jnp.inf

    total_size = jax.tree.leaves(genotypes)[0].shape[0]
    num_batches = total_size // eval_batch_size

    # Reshape genotypes to (num_batches, eval_batch_size, ...)
    reshaped_genotypes = jax.tree.map(
        lambda x: x.reshape((num_batches, eval_batch_size) + x.shape[1:]),
        genotypes,
    )

    def scan_eval_batch(carry, batch_data):
        key = carry
        batch_geno, batch_idx = batch_data
        key, subkey = jax.random.split(key)
        batch_fit, batch_desc, batch_extra = scoring_fn(batch_geno, subkey)
        return key, (batch_fit, batch_desc, batch_extra)

    batch_indices = jnp.arange(num_batches)
    key, (all_fitnesses, all_descriptors, all_extra_scores) = jax.lax.scan(
        scan_eval_batch,
        key,
        (reshaped_genotypes, batch_indices),
        length=num_batches,
    )

    fitnesses = all_fitnesses.reshape(-1)
    descriptors = all_descriptors.reshape(total_size, -1)

    if all_extra_scores:
        extra_scores = jax.tree.map(
            lambda x: x.reshape((total_size,) + x.shape[2:]) if x.ndim > 2 else x.reshape(-1),
            all_extra_scores,
        )
    else:
        extra_scores = {}

    fitnesses = jnp.reshape(fitnesses, (fitnesses.shape[0], 1))

    valid_fitnesses = jnp.where(
        valid_mask[:, None],
        fitnesses,
        -jnp.inf * jnp.ones_like(fitnesses),
    )

    temp_dns = DominatedNoveltySearch(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        population_size=population_size,
        k=k,
    )

    new_repertoire, _, _ = temp_dns.init_ask_tell(
        genotypes=genotypes,
        fitnesses=valid_fitnesses,
        descriptors=descriptors,
        key=key,
        extra_scores=extra_scores,
    )

    return new_repertoire, key


# -- Core continual training function ----------------------------------------

def train_dns_continual(
    *,
    pop_size: int = 256,
    batch_size: int = 128,
    generations_per_task: int = 200,
    iso_sigma: float = 0.05,
    line_sigma: float = 0.5,
    k: int = 3,
    num_evals: int = 3,
    episode_length: int = 1000,
    eval_batch_size: int = 128,
    seed: int = 0,
    trial_idx: int = 1,
    project_dir: str | None = None,
    use_wandb: bool = True,
    wandb_project: str = "Kinetix-continual-dns",
    log_interval: int = 10,
):
    """Continual DNS training loop across all Kinetix environments using QDax."""
    rng = jr.PRNGKey(seed)

    # -- load all 20 envs up-front --------------------------------------------
    base_yaml = load_base_config()
    print("Loading all environments...")
    envs_data = []
    for ename in ENVIRONMENTS:
        print(f"  Loading {ename}...")
        envs_data.append(load_env(ename, base_yaml))
    print(f"  All {len(ENVIRONMENTS)} environments loaded.\n")

    # Use first env to initialise network (same architecture for all)
    config0, env0, init_es0, static_ep0, ep0 = envs_data[0]
    network = make_network_from_config(env0, ep0, config0, actor_only=True)

    rng, init_rng = jr.split(rng)
    dummy_obs, _ = jax.vmap(env0.reset, (0, None))(jr.split(init_rng, 1), ep0)
    dones = jnp.zeros(1, dtype=jnp.bool_)
    init_hstate = ScannedRNN.initialize_carry(1)
    init_x = jax.tree.map(lambda x: x[None, ...], (dummy_obs, dones))

    rng, param_rng = jr.split(rng)
    base_network_params = network.init(param_rng, init_hstate, init_x)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(base_network_params))
    print(f"Actor-only network param count: {param_count}")

    total_generations = generations_per_task * len(ENVIRONMENTS)

    # -- project dir setup ----------------------------------------------------
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

    print(f"\n=== Kinetix DNS Continual Training (QDax) ===")
    print(f"  Trial: {trial_idx}")
    print(f"  Seed: {seed}")
    print(f"  Population size: {pop_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Generations per task: {generations_per_task}")
    print(f"  Total generations: {total_generations}")
    print(f"  Episode length: {episode_length}")
    print(f"  Num evals (rollouts averaged): {num_evals}")
    print(f"  Eval batch size (re-eval chunking): {eval_batch_size}")
    print(f"  iso_sigma: {iso_sigma}")
    print(f"  line_sigma: {line_sigma}")
    print(f"  k (novelty neighbors): {k}")
    print(f"  Param count: {param_count}")
    print(f"  Num tasks: {len(ENVIRONMENTS)}")
    if output_dir:
        print(f"  Output directory: {output_dir}")
    print(f"=============================================\n")

    # -- wandb ----------------------------------------------------------------
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=f"DNS_continual_trial{trial_idx}_seed{seed}",
            config={
                "pop_size": pop_size,
                "batch_size": batch_size,
                "generations_per_task": generations_per_task,
                "total_generations": total_generations,
                "episode_length": episode_length,
                "num_evals": num_evals,
                "eval_batch_size": eval_batch_size,
                "iso_sigma": iso_sigma,
                "line_sigma": line_sigma,
                "k": k,
                "seed": seed,
                "trial_idx": trial_idx,
                "param_count": param_count,
                "optimizer": "DNS_QDax",
                "num_tasks": len(ENVIRONMENTS),
                "continual": True,
            },
        )

    # -- Create scoring functions for every env up-front ----------------------
    scoring_fns = {}
    scoring_fns_lightweight = {}
    for tidx, (cfg_t, env_t, init_es_t, static_ep_t, ep_t) in enumerate(envs_data):
        ename = ENVIRONMENTS[tidx]
        play_step = create_play_step_fn(env_t, ep_t, network)
        env_reset = create_env_reset_fn(env_t, ep_t, init_es_t)
        scoring_fns[ename] = functools.partial(
            scoring_function_kinetix,
            episode_length=episode_length,
            play_reset_fn=env_reset,
            play_step_fn=play_step,
            descriptor_extractor=descriptor_extractor_fn,
            num_evals=num_evals,
        )
        scoring_fns_lightweight[ename] = functools.partial(
            scoring_function_kinetix_lightweight,
            episode_length=episode_length,
            play_reset_fn=env_reset,
            play_step_fn=play_step,
            descriptor_extractor=descriptor_extractor_fn,
            num_evals=num_evals,
        )

    # -- QDax emitter ---------------------------------------------------------
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )

    metrics_function = functools.partial(default_qd_metrics, qd_offset=0.0)

    # -- Initialise population ------------------------------------------------
    rng, pop_rng = jr.split(rng)
    init_rnn_state = ScannedRNN.initialize_carry(1)
    if isinstance(init_rnn_state, tuple):
        init_rnn_state = init_rnn_state[0]

    keys = jr.split(pop_rng, batch_size)

    def create_initial_policy(key):
        return PolicyState(
            network_params=base_network_params,
            rnn_state=init_rnn_state,
            n_dormant=jnp.array([0.0]),
        )

    init_variables = jax.vmap(create_initial_policy)(keys)

    # -- GIF helper (closure over `network`) ----------------------------------
    def save_task_gifs(
        task_idx, env_name, config_t, env_t, ep_t, static_ep_t,
        init_env_state, best_policy_state, num_gifs=10,
    ):
        if output_dir is None:
            return
        task_gifs = os.path.join(output_dir, "gifs", f"task{task_idx}_{env_name}")
        os.makedirs(task_gifs, exist_ok=True)

        eval_env = make_kinetix_env(
            observation_type=config_t["observation_type"],
            action_type=config_t["action_type"],
            reset_fn=make_reset_fn_from_config(config_t, ep_t, static_ep_t),
            static_env_params=static_ep_t,
        )
        render_sep = eval_env.static_env_params.replace(downscale=4)
        pixel_renderer = jax.jit(make_render_pixels(ep_t, render_sep))

        @jax.jit
        def get_action(params, hstate, obs_batched, done, rng_key):
            ac_in = jax.tree.map(lambda x: x[None, ...], (obs_batched, done))
            new_hstate, pi = network.apply(params, hstate, ac_in)
            action = pi.sample(seed=rng_key).squeeze(0)
            return action, new_hstate

        @jax.jit
        def env_step_jit(rng_key, state, action):
            return eval_env.step(rng_key, state, action, ep_t)

        @jax.jit
        def env_reset_jit(rng_key):
            return eval_env.reset(rng_key, ep_t, override_reset_state=init_env_state)

        for gif_idx in range(num_gifs):
            eval_rng = jr.PRNGKey(seed * 1000 + task_idx * 100 + gif_idx)
            obs, env_state = env_reset_jit(eval_rng)
            hstate = ScannedRNN.initialize_carry(1)
            done = jnp.zeros(1, dtype=jnp.bool_)
            frames = []
            total_reward = 0.0

            for step in range(ep_t.max_timesteps):
                frame = np.array(pixel_renderer(env_state))
                frame = frame.transpose(1, 0, 2)[::-1].astype(np.uint8)
                frames.append(frame)

                eval_rng, act_rng = jr.split(eval_rng)
                obs_batched = jax.tree.map(lambda x: x[None, ...], obs)
                action, hstate = get_action(
                    best_policy_state.network_params, hstate, obs_batched, done, act_rng,
                )
                action = action.squeeze(0) if hasattr(action, "squeeze") else action[0]

                eval_rng, step_rng = jr.split(eval_rng)
                obs, env_state, reward, step_done, info = env_step_jit(
                    step_rng, env_state, action,
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
        print(f"    Saved {num_gifs} GIFs for task {task_idx} ({env_name}) -> {task_gifs}")

    # =========================================================================
    # Training loop across tasks
    # =========================================================================
    print("Starting continual DNS training loop...")
    start_time = time.time()
    global_gen = 0
    best_fitness_ever = -float("inf")

    # Initialise DNS with first environment
    first_env_name = ENVIRONMENTS[0]
    current_scoring_fn = scoring_fns[first_env_name]

    dns = DominatedNoveltySearch(
        scoring_function=current_scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        population_size=pop_size,
        k=k,
    )

    rng, subkey = jr.split(rng)
    repertoire, emitter_state, init_metrics = dns.init(init_variables, subkey)

    # Set up metrics tracking
    log_period = 1
    metrics_keys_list = [
        "iteration", "qd_score", "coverage", "max_fitness", "time", "current_env",
    ]
    metrics = {mk: jnp.array([]) for mk in metrics_keys_list}

    init_metrics = jax.tree.map(
        lambda x: jnp.array([x]) if x.shape == () else x, init_metrics
    )
    init_metrics["iteration"] = jnp.array([0], dtype=jnp.int32)
    init_metrics["time"] = jnp.array([0.0])
    init_metrics["current_env"] = jnp.array([0], dtype=jnp.int32)

    metrics = jax.tree.map(
        lambda m, im: jnp.concatenate([m, im], axis=0), metrics, init_metrics
    )

    csv_path = (
        os.path.join(output_dir, "dns_continual_logs.csv")
        if output_dir
        else f"dns_continual_trial{trial_idx}_logs.csv"
    )
    csv_logger = CSVLogger(csv_path, header=list(metrics.keys()))
    csv_logger.log(jax.tree.map(lambda x: x[-1], metrics))

    # -- iterate over tasks ---------------------------------------------------
    for task_idx, env_name in enumerate(ENVIRONMENTS):
        config_t, env_t, init_es_t, static_ep_t, ep_t = envs_data[task_idx]
        bare = env_name.split("/")[-1] if "/" in env_name else env_name

        print(f"\n{'='*60}")
        print(
            f"Task {task_idx}/{len(ENVIRONMENTS)-1}: {bare}  "
            f"(gens {global_gen}..{global_gen + generations_per_task - 1})"
        )
        print(f"{'='*60}")

        current_scoring_fn = scoring_fns[env_name]
        current_scoring_fn_lw = scoring_fns_lightweight[env_name]

        if task_idx > 0:
            # Re-evaluate repertoire on the new task
            print(f"  Re-evaluating repertoire on task {bare}...")
            repertoire, rng = reevaluate_repertoire(
                repertoire,
                current_scoring_fn_lw,
                rng,
                mixing_emitter,
                metrics_function,
                pop_size,
                k,
                eval_batch_size=eval_batch_size,
            )

        # (Re)create DNS with current scoring function
        dns = DominatedNoveltySearch(
            scoring_function=current_scoring_fn,
            emitter=mixing_emitter,
            metrics_function=metrics_function,
            population_size=pop_size,
            k=k,
        )
        dns_scan_update = dns.scan_update

        # (Re)initialise emitter state
        rng, subkey = jr.split(rng)
        emitter_state = mixing_emitter.init(
            key=subkey,
            repertoire=repertoire,
            genotypes=repertoire.genotypes,
            fitnesses=repertoire.fitnesses,
            descriptors=repertoire.descriptors,
            extra_scores=repertoire.extra_scores,
        )

        # Run generations for this task
        local_gen = 0
        while local_gen < generations_per_task:
            batch = min(log_period, generations_per_task - local_gen)
            if batch <= 0:
                break

            t0 = time.time()
            (repertoire, emitter_state, rng), current_metrics = jax.lax.scan(
                dns_scan_update,
                (repertoire, emitter_state, rng),
                (),
                length=batch,
            )
            dt = time.time() - t0

            current_metrics["iteration"] = jnp.arange(
                global_gen + 1, global_gen + batch + 1, dtype=jnp.int32
            )
            current_metrics["time"] = jnp.repeat(dt / batch, batch)
            current_metrics["current_env"] = jnp.full(batch, task_idx, dtype=jnp.int32)

            metrics = jax.tree.map(
                lambda m, cm: jnp.concatenate([m, cm], axis=0), metrics, current_metrics
            )
            csv_logger.log(jax.tree.map(lambda x: x[-1], metrics))

            # Wandb logging
            gen_best = float(metrics["max_fitness"][-1])
            gen_qd = float(metrics["qd_score"][-1])
            gen_cov = float(metrics["coverage"][-1])

            if gen_best > best_fitness_ever:
                best_fitness_ever = gen_best

            if use_wandb:
                wandb.log(
                    {
                        "generation": global_gen + batch,
                        "task": task_idx,
                        "task_name": bare,
                        "best_fitness": gen_best,
                        "qd_score": gen_qd,
                        "coverage": gen_cov,
                        "best_overall": best_fitness_ever,
                    },
                    step=global_gen + batch,
                )

            elapsed = time.time() - start_time
            if local_gen % log_interval == 0 or local_gen + batch >= generations_per_task:
                print(
                    f"Gen {global_gen + batch:4d} (task {task_idx} local {local_gen + batch:3d})  "
                    f"best={gen_best:8.2f}  qd={gen_qd:8.2f}  "
                    f"cov={gen_cov:.0f}  overall={best_fitness_ever:8.2f}  "
                    f"time={elapsed:6.1f}s"
                )

            global_gen += batch
            local_gen += batch

        # -- end of task: save GIFs -------------------------------------------
        best_idx = jnp.argmax(repertoire.fitnesses)
        best_ps = jax.tree.map(lambda x: x[best_idx], repertoire.genotypes)
        task_best_fitness = float(repertoire.fitnesses[best_idx, 0])

        print(f"  Task {task_idx} ({bare}) finished.  best={task_best_fitness:.2f}")
        save_task_gifs(
            task_idx, bare, config_t, env_t, ep_t, static_ep_t, init_es_t, best_ps,
        )

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Continual DNS training complete! Total time: {total_time:.1f}s")
    print(f"Best fitness ever: {best_fitness_ever:.2f}")
    print(f"{'='*60}")

    # -- save checkpoint ------------------------------------------------------
    if output_dir:
        best_idx = jnp.argmax(repertoire.fitnesses)
        best_policy = jax.tree.map(lambda x: x[best_idx], repertoire.genotypes)
        ckpt_path = os.path.join(output_dir, "dns_continual_best.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump(
                {
                    "params": jax.tree.map(np.array, best_policy.network_params),
                    "policy_state": jax.tree.map(np.array, best_policy),
                    "best_fitness": float(best_fitness_ever),
                    "total_generations": total_generations,
                    "generations_per_task": generations_per_task,
                    "seed": seed,
                    "pop_size": pop_size,
                    "batch_size": batch_size,
                    "iso_sigma": iso_sigma,
                    "line_sigma": line_sigma,
                    "k": k,
                    "num_evals": num_evals,
                    "eval_batch_size": eval_batch_size,
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
                    "pop_size": pop_size,
                    "batch_size": batch_size,
                    "iso_sigma": iso_sigma,
                    "line_sigma": line_sigma,
                    "k": k,
                    "num_evals": num_evals,
                    "eval_batch_size": eval_batch_size,
                    "seed": seed,
                    "trial_idx": trial_idx,
                    "param_count": param_count,
                    "num_tasks": len(ENVIRONMENTS),
                },
                f,
                indent=2,
            )
        print(f"  Saved metrics: {metrics_path}")

    # -- cleanup --------------------------------------------------------------
    if use_wandb:
        wandb.finish()
    if tee_logger:
        sys.stdout = tee_logger.stdout
        tee_logger.close()

    # Free memory
    try:
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()

    return best_fitness_ever


# -- CLI ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Continual DNS (QDax) on Kinetix – sequential tasks, no population reset"
    )
    parser.add_argument("--gpu", type=str, default=None, help="GPU device ID")
    parser.add_argument("--pop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--generations_per_task", type=int, default=200)
    parser.add_argument("--iso_sigma", type=float, default=0.05)
    parser.add_argument("--line_sigma", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=3, help="k nearest neighbors for novelty")
    parser.add_argument("--num_evals", type=int, default=3,
                        help="Number of rollouts averaged for fitness (default 3)")
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=128,
                        help="Chunk size for batched re-evaluation at task switch (default 128)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Kinetix-continual-dns")
    parser.add_argument(
        "--project_dir",
        type=str,
        default=None,
        help="Root project dir for structured output (e.g. projects/kinetix)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for trial in range(1, args.num_trials + 1):
        print(f"\n{'#'*60}")
        print(f"# Continual DNS (QDax)  Trial: {trial}")
        print(f"{'#'*60}")
        train_dns_continual(
            pop_size=args.pop_size,
            batch_size=args.batch_size,
            generations_per_task=args.generations_per_task,
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
