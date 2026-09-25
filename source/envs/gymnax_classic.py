"""The gymnax task family, its policy, and how a genome is scored on it.

A *task* is a gymnax environment plus a fixed observation-offset vector: the
policy sees ``obs + noise_vector`` instead of ``obs``. Sub-task 0 has a zero
offset and is the unperturbed environment; every later sub-task adds a vector
drawn once from ``N(0, noise_range^2 I)``.

Since 2026-09-05 a sub-task can instead be a change to the PHYSICS
(``task_mod: physics``, selected with ``--task_options task_mod=physics``):
the vector is one number, a multiplier on a named parameter of the gymnax
``EnvParams`` -- the pole length on CartPole, the link masses on Acrobot, the
engine force on MountainCar -- and the observation is untouched. That is the
construction the generalisation literature uses for these three environments
(Packer et al. 2018, "Assessing Generalization in Deep Reinforcement
Learning", varies exactly these: CartPole force / length / pole mass, Acrobot
length / mass / moment of inertia, MountainCar force and gravity) and the
gymnax counterpart of the ant's friction sub-tasks in ``tasks_mjx.py``. It is
wired the same way: ``TaskSpec`` is the ``env_params`` slot, the multiplier is
a TRACED argument applied to the params inside the rollout, so a switching run
compiles once, and the two-sub-task experiment is the unperturbed environment
against one rescaled one. ``PHYSICS_PARAMS`` says what each name rescales, and
``ENV_CONFIGS[env]['physics']`` the default name and multipliers.

This construction, the offset draw, the policy architecture and the rollout are
taken unchanged from this repo's own continual study
(``source/studies/gymnax/continual_common.py``, ``source/algorithms/networks.py``).
That is deliberate and load-bearing: this project's claims are meant to sit next
to that study's plasticity numbers, and they only do so if the tasks are the
same tasks. In particular the offset stream is seeded from the *trial index*
alone and never from the training RNG, which buys two things:

  - Two methods at the same trial face exactly the same sub-tasks.
  - The first n sub-tasks of a long run are the first n of a short one, so the
    two-task experiment here is literally the first two sub-tasks of the
    ten-task experiment that follows it.

An offset is not a cosmetic perturbation. CartPole terminates when the pole
passes 0.2095 rad, so an offset of order 1 rad in the angle coordinate puts the
policy's decision boundary in a completely different place in state space; the
optimum moves, and whether the two optima share a basin is the open question.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import random, flatten_util

from source.algorithms.networks import MLPPolicy  # noqa: F401  (re-exported)
from source.utils.task_sequence import (  # noqa: F401  (cycle re-exported)
    GYMNAX_PHYSICS_TASKS, action_flip_sequence, cycle_task_sequence,
    physics_mult_sequence)
from source.metrics.aurora import (  # noqa: F401  (re-exported)
    episode_relative_indices,
)

def build_policy(key, obs_dim, action_dim, hidden_dims):
    """Return (policy, param_template, num_params)."""
    policy = MLPPolicy(hidden_dims=tuple(hidden_dims), action_dim=action_dim)
    param_template = policy.init(key, jnp.zeros((obs_dim,)))
    num_params = flatten_util.ravel_pytree(param_template)[0].shape[0]
    return policy, param_template, int(num_params)


def unflatten_params(flat_params, param_template):
    _, unravel_fn = flatten_util.ravel_pytree(param_template)
    return unravel_fn(flat_params)


def _task_noise_vector_list(trial, num_tasks, obs_dim, noise_range,
                       first_task_clean=True):
    """The sub-task sequence for a trial.

    Sub-task 0 is the unperturbed environment; every later sub-task adds a
    fixed observation-noise vector. The stream is seeded from the trial index
    alone, deliberately independent of the training RNG, which has two
    consequences worth relying on:

      - Different methods at the same trial face exactly the same sub-tasks, so
        they are comparable run for run.
      - The first n sub-tasks of a long run are identical to those of a short
        one, so a 20-sub-task run also contains the 10-sub-task experiment.

    ``first_task_clean=False`` draws sub-task 0 as well, giving a sequence that
    is symmetric about the unperturbed environment rather than anchored to it.
    It costs the nesting property in the second bullet above, so it is off by
    default and every run of the benchmarking paper uses the default. The
    generalists study passes False for its two-task landscapes; see
    `source/envs/gymnax_classic.py`.
    """
    rng = jax.random.key(int(trial) * 7919)
    vectors = []
    for i in range(num_tasks):
        if i == 0 and first_task_clean:
            vectors.append(jnp.zeros((obs_dim,)))
            continue
        rng, noise_key = random.split(rng)
        vectors.append(random.normal(noise_key, (obs_dim,)) * noise_range)
    return vectors


def make_episode_fn(env, policy, param_template, episode_length):
    """Return ``episode(flat_params, key, noise_vector, env_params) -> return``.

    One undiscounted episode return. Reward accrues only until the first
    termination, but the scan always runs the full ``episode_length`` so the
    computation stays a fixed shape and vmaps cleanly.
    """

    def episode(flat_params, key, noise_vector, env_params):
        params = unflatten_params(flat_params, param_template)
        obs, state = env.reset(key, env_params)

        def step_fn(carry, _):
            obs, state, total_reward, done_flag, key = carry
            logits = policy.apply(params, obs + noise_vector)
            action = jnp.argmax(logits)
            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(
                step_key, state, action, env_params
            )
            total_reward = total_reward + reward * (1.0 - done_flag)
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
            return (next_obs, next_state, total_reward, done_flag, key), None

        (_, _, total_reward, _, _), _ = jax.lax.scan(
            step_fn, (obs, state, 0.0, 0.0, key), None, length=episode_length
        )
        return total_reward

    return episode


# The sub-task construction, the policy and the rollout are this repo's own,
# imported rather than copied: the study's claims only sit next to the
# benchmarking paper's plasticity numbers if the tasks are literally the same
# tasks, and an import is the only way to make that true by construction.
# `scripts/outdated/generalists/check_frozen.py` pins the values these produce, so a
# change to them on the paper's side fails loudly instead of silently moving
# this study's numbers.


# What counts as solving each environment. Taken verbatim from the earlier
# study (`source/metrics/evaluation_metrics.py`), which keeps three
# named sets because it contains runs reported against all three and they are
# NOT interchangeable -- on Acrobot the "repo" value of -70 sits past what its
# gymnax neuroevolution runs reach at any population size, so every cell floors
# at zero and a real trend becomes invisible. Report the set you used.
#
# One caveat from that file does not carry over. Its "rebuttal" set is loosened
# partly because NE there logs `best_fitness`, a max over 512 population
# members, while RL logs one policy -- a gap of ~8 reward on Acrobot and ~45 on
# MountainCar. Nothing in this project logs a max over a population: the
# generalist score is always ONE deterministic policy (the NES centroid, or
# PPO's actor) under argmax actions. So that half of the rationale is absent
# here and only the 500-step episode cap applies.
#
# The threshold is written into every run's results.json, and the analysis
# scripts take a `--threshold` override, so a set can be changed after the fact
# without re-running anything.
# DeepSea (bsuite, through gymnax): `DeepSea<N>-bsuite` is the N x N grid.
# The wrapper, the sub-task construction and why it is here: see DeepSeaEnv.
DEEPSEA_SIZES = (8, 10, 12, 14, 16, 20)
DEEPSEA_ENV_NAMES = tuple(f"DeepSea{n}-bsuite" for n in DEEPSEA_SIZES)


def deepsea_size(env_name):
    """Grid size of a ``DeepSea<N>-bsuite`` name, else None."""
    m = re.fullmatch(r"DeepSea(\d+)-bsuite", str(env_name))
    return int(m.group(1)) if m else None


THRESHOLD_SETS = {
    "repo": {
        "CartPole-v1": 475.0,
        "Acrobot-v1": -70.0,
        "MountainCar-v0": -110.0,
    },
    "pbt_doc": {
        "CartPole-v1": 475.0,
        "Acrobot-v1": -90.0,
        "MountainCar-v0": -120.0,
    },
    "rebuttal": {
        "CartPole-v1": 400.0,
        "Acrobot-v1": -120.0,
        "MountainCar-v0": -200.0,
    },
    # The set this project reports against, chosen from what the methods
    # measurably reach on the UNPERTURBED task rather than from convention.
    # A sigma x learning-rate sweep on the unperturbed task produced them.
    #
    #   CartPole    475, unchanged -- both methods reach exactly 500, so the
    #               usual threshold separates cleanly and nothing is gained by
    #               moving it. Every CartPole result already on disk was scored
    #               at 475 and is unaffected by this set existing.
    #   Acrobot     -80 rather than the repo's -70. NES solves the unperturbed
    #               task at -64, but its *generalist* scores sit at -71 to -75:
    #               a policy competent on BOTH sub-tasks lands just the wrong
    #               side of -70, so that threshold would report "generalist
    #               never found" for exactly the agents this project is looking
    #               for. -80 sits below the achievable generalist band and well
    #               above the ~-500 a policy that fails one sub-task scores.
    #   MountainCar -150 rather than -110. Tuned NES reaches about -127 on the
    #               unperturbed task, so -110 is not attainable here and would
    #               floor every cell at zero; -150 is comfortably reachable and
    #               still far from the -500 floor.
    #
    # Neither loosening is free: a looser threshold makes it easier for a
    # transiently-lucky specialist to be counted, which is why the transfer-hard
    # test and the retention column matter more on these two environments than
    # they do on CartPole.
    "generalists": {
        "CartPole-v1": 475.0,
        "Acrobot-v1": -80.0,
        "MountainCar-v0": -150.0,
    },
}

# Reaching the treasure pays 1.0 and anything else 0.0, under every set.
for _set in THRESHOLD_SETS.values():
    _set.update({_name: 0.5 for _name in DEEPSEA_ENV_NAMES})

DEFAULT_THRESHOLD_SET = "generalists"

# Per-environment settings. `hidden_dims` is pinned to what every method in the
# earlier gymnax study searched -- changing it makes runs here incomparable
# with that study's, which is the whole reason the number is written down.
ENV_CONFIGS = {
    "CartPole-v1": {
        "hidden_dims": (16, 16),
        "episode_length": 500,
        # The offset sigma this environment's runs are made at. Stated rather
        # than left to a caller default because it does NOT transfer between
        # bodies -- the mjx suite's is a different number for the same reason.
        "noise_range": 1.0,
        "solved_threshold": THRESHOLD_SETS[DEFAULT_THRESHOLD_SET]["CartPole-v1"],
        # What a sub-task IS by default (every run made before 2026-09-05
        # is obs_noise), and the physics variant's knob when selected:
        # `param` names a group in PHYSICS_PARAMS and `mults[i]` is what
        # sub-task i multiplies it by (cycled past the end). The two-sub-task
        # experiment is therefore the stock body against `mults[1]`.
        #
        # Chosen on 2026-09-05 from `analysis/physics_zero_shot.py`, which
        # scores section C's NES specialists (trained on the stock body)
        # under every group at 0.25x-4x. A shift is only a sub-task if the
        # stock specialist does NOT already solve it. On CartPole the pole
        # mass, cart mass and force change nothing at any multiplier (8/8
        # specialists at 500 from 0.25x to 4x); a pole twice as long drops
        # all eight to 38.6, a quarter as long to 284, and gravity only at 4x.
        "task_mod": "obs_noise",
        "physics": {"param": "length", "mults": [1.0, 2.0]},
    },
    "Acrobot-v1": {
        "hidden_dims": (16, 16),
        "episode_length": 500,
        # The offset sigma this environment's runs are made at. Stated rather
        # than left to a caller default because it does NOT transfer between
        # bodies -- the mjx suite's is a different number for the same reason.
        "noise_range": 1.0,
        "solved_threshold": THRESHOLD_SETS[DEFAULT_THRESHOLD_SET]["Acrobot-v1"],
        # Link masses 1.15x: the stock specialists fall from -68.9 (8/8 at
        # the -80 threshold) to -93.6 (0/8), and a NES specialist trained on
        # it reaches -74.9, so the threshold is attainable there -- barely,
        # and that is the constraint. At 1.25x a specialist ends at -85.5, at
        # 1.5x at -99.4 (best ever -82.8), at 2x at -138: every larger shift
        # puts the -80 threshold out of reach and voids found/held. Length
        # and moment of inertia at 1.15x fail the stock specialists the same
        # way (-93.3 and -92.3, 0/8). Halving the masses is bimodal (4/8).
        "task_mod": "obs_noise",
        "physics": {"param": "mass", "mults": [1.0, 1.15]},
    },
    "MountainCar-v0": {
        "hidden_dims": (16, 16),
        "episode_length": 500,
        # The offset sigma this environment's runs are made at. Stated rather
        # than left to a caller default because it does NOT transfer between
        # bodies -- the mjx suite's is a different number for the same reason.
        "noise_range": 1.0,
        "solved_threshold": THRESHOLD_SETS[DEFAULT_THRESHOLD_SET]["MountainCar-v0"],
        # Gravity 1.5x: the stock specialists fall from -105.6 (4/4 at the
        # -150 threshold) to -193.8 (1/4), and a specialist trained on it
        # reaches -126, so the threshold is attainable there. Engine force
        # 0.75x is the same shift from the other side (-200.9 zero-shot) but
        # its specialist ends AT the threshold (-152.8, best -135.5), which
        # would make found/held a coin toss; 0.5x is out of reach (-222).
        # More force or less gravity transfers zero-shot everywhere (4/4)
        # and would be no sub-task.
        "task_mod": "obs_noise",
        "physics": {"param": "gravity", "mults": [1.0, 1.5]},
    },
}

for _n in DEEPSEA_SIZES:
    ENV_CONFIGS[f"DeepSea{_n}-bsuite"] = {
        "hidden_dims": (16, 16),
        # The agent descends one row a step, so an episode is exactly N steps.
        "episode_length": _n,
        # The observation is a one-hot cell; an offset on it means nothing,
        # and the family this environment is for is `actions` (the map).
        "noise_range": 0.0,
        "solved_threshold": THRESHOLD_SETS[DEFAULT_THRESHOLD_SET][f"DeepSea{_n}-bsuite"],
        "task_mod": "actions",
    }

TASK_MODS = ('obs_noise', 'physics', 'actions')

# Which `EnvParams` fields a physics multiplier rescales, by environment and by
# the name a caller uses for it. Every entry is a group rather than a single
# field because gymnax stores DERIVED quantities as their own fields and the
# dynamics read those: CartPole's `total_mass` and `polemass_length` are
# `masscart + masspole` and `masspole * length`, so rescaling `length` alone
# would leave the pole's moment term at the old length and the result would be
# a body no physics describes. `apply_physics` recomputes CartPole's two
# derived fields from the rescaled primitives whatever was rescaled, and scales
# Acrobot's link centre-of-mass positions with its link lengths so the links
# stay uniform rods. The names are Packer et al. 2018's parameters where that
# paper has one (`length`, `masspole`, `force_mag` on CartPole; `length`,
# `mass`, `moi` on Acrobot; `force` and `gravity` on MountainCar) plus
# `gravity` and `masscart` on CartPole, which are the other two knobs a
# CartPole has.
PHYSICS_PARAMS = {
    "CartPole-v1": {
        "length":    ("length",),
        "masspole":  ("masspole",),
        "masscart":  ("masscart",),
        "gravity":   ("gravity",),
        "force_mag": ("force_mag",),
    },
    "Acrobot-v1": {
        "mass":   ("link_mass_1", "link_mass_2"),
        "length": ("link_length_1", "link_length_2",
                   "link_com_pos_1", "link_com_pos_2"),
        "moi":    ("link_moi",),
    },
    "MountainCar-v0": {
        "force":         ("force",),
        "gravity":       ("gravity",),
        # The other two knobs CARL (Benjamins et al. 2023) contextualises on
        # this body; neither is a Packer et al. parameter. Added 2026-09-07
        # for the pilots that asked whether any MountainCar rescaling gives
        # two sub-tasks that are not nested.
        "max_speed":     ("max_speed",),
        "goal_position": ("goal_position",),
    },
}


def apply_physics(env_name, base, param, mult):
    """``base`` with the fields of ``param`` multiplied by ``mult``.

    ``mult`` may be a traced scalar: gymnax's ``EnvParams`` is a flax struct
    whose fields are pytree leaves, so the result is a params pytree the jitted
    rollout can take as an argument, which is what lets the sub-task switch
    without a recompile. CartPole's derived fields are recomputed from the
    (possibly rescaled) primitives so the body stays self-consistent; see
    PHYSICS_PARAMS.
    """
    fields = PHYSICS_PARAMS[env_name][param]
    changes = {f: getattr(base, f) * mult for f in fields}
    params = base.replace(**changes)
    if 'CartPole' in env_name:
        params = params.replace(
            total_mass=params.masscart + params.masspole,
            polemass_length=params.masspole * params.length)
    return params


# ---------------------------------------------------------------------------
# Reversed controls: a sub-task in the ACTION map (`task_mod: actions`)
# ---------------------------------------------------------------------------
#
# Every physics rescaling of MountainCar came out NESTED on 2026-09-07: a
# policy that pumps the swing harder solves every weaker car, and the only
# non-transferring policies were the push-right traps on stronger ones. The
# reason is that this body's control law, "push along the velocity", is
# physics-agnostic. What conflicts with it is a change to what the right
# action IS: sub-task 1 reverses the action order, so the push that pumps
# energy on the stock body damps it on the reversed one. Same physics, same
# observation.
#
# For a memoryless policy of the observation -- the NE MLP and PPO's actor
# alike -- that pair may have NO generalist: the two sub-tasks want opposite
# outputs at the same input. `cue` adds section C's observation offset to
# sub-task 1 as well (the trial's own draw at `noise_range`), which is what
# makes the sub-task identifiable from the observation and a generalist
# possible. A row of `task_vectors` under this mode is
# `[flip, offset_0, ..., offset_{obs_dim-1}]`; the offset is zero without
# `cue`.

@struct.dataclass
class FlippedParams:
    """gymnax params plus a traced flag: reverse the action order or not."""
    base: object
    flip: jnp.ndarray


class FlipEnv:
    """A gymnax env whose ``step`` reverses the action under ``FlippedParams``.

    ``reset``/``step`` take a ``FlippedParams`` (or bare params, passed
    through unchanged) so that ``make_episode_fn`` and every scoring function
    here run on it without modification: the action reaches the underlying
    env as ``n - 1 - a`` when ``flip > 0.5`` and as ``a`` otherwise, and
    ``flip`` is a traced scalar, so a switching run compiles once.
    """

    def __init__(self, env):
        self.env = env
        self.num_actions = int(env.action_space(env.default_params).n)

    @staticmethod
    def _split(params):
        if isinstance(params, FlippedParams):
            return params.base, params.flip
        return params, None

    def reset(self, key, params):
        base, _ = self._split(params)
        return self.env.reset(key, base)

    def step(self, key, state, action, params):
        base, flip = self._split(params)
        if flip is not None:
            action = jnp.where(flip > 0.5, self.num_actions - 1 - action,
                               action)
        return self.env.step(key, state, action, base)

    def __getattr__(self, item):        # observation_space, action_space, ...
        return getattr(self.env, item)


# ---------------------------------------------------------------------------
# DeepSea: a sparse grid whose sub-task is the ACTION MAP
# ---------------------------------------------------------------------------
#
# bsuite's DeepSea (Osband et al. 2019), gymnax's JAX port. The agent starts
# at the top-left of an N x N grid and descends one row a step; in every cell
# one of the two actions moves it right and the other left, and only the
# bottom-right cell pays, +1 on the last step. A policy that does not reach it
# scores exactly 0 whatever it does, so the fitness landscape is flat except
# on the one N-cell solving path. Which action means "right" in each cell is
# the ACTION MAP, and that map is what a sub-task changes here: sub-task 0 is
# the stock map (action 1 is "right" everywhere), and a sub-task with flag
# f > 0 draws a Bernoulli map from a key folded with f, so about half the
# cells on the solving path change meaning at every switch.
#
# Why this is the collapse task. After a switch the incumbent's path hits a
# remapped cell within a few rows, and a converged population -- every member
# on nearly the same path -- scores 0 to a member: the whole population is at
# the floor with nothing to select on, which is the picture MountainCar's
# action reversal gives at -500. A population that still spans many paths has
# members deeper in the grid under the new map, and novelty on the cells
# reached points at the one cell nobody has visited.
#
# gymnax's move cost (0.01 a right move, "unscaled_move_cost") is ZEROED here.
# With it, non-solvers are ranked by how few right moves they make, so
# truncation selection drifts a population to the left wall -- the deceptive
# DeepSea of the exploration literature, on which a GA fails even on the stock
# map. Zero cost makes every non-solver tie, so sub-task 0 is reachable by
# neutral drift and it is the switch, not the task, that floors the population.
#
# The map rides in ``FlippedParams.flip``, the traced scalar the reversal
# family already threads through every scoring function: a switching run
# compiles once, nothing in the rollouts changes, and the post-hoc passes that
# rebuild the environment from the saved ``action_flips`` get the same maps
# back. ``action_flip_sequence`` gives DeepSea one flag per distinct sub-task
# (folded with the trial) instead of the alternating 0/1.

DEEPSEA_MAP_KEY = 4241          # base key; a sub-task's flag is folded into it


class DeepSeaEnv:
    """gymnax's DeepSea with a flat observation and a per-sub-task action map.

    ``reset``/``step`` take a ``FlippedParams`` (or bare params, meaning the
    stock map) exactly as ``FlipEnv`` does. The (N, N) one-hot observation is
    flattened so the classic-control MLP reads it unchanged.
    """

    def __init__(self, env):
        self.env = env
        self.size = int(env.size)
        self.num_actions = 2

    def _action_map(self, flip):
        stock = jnp.ones((self.size, self.size), dtype=jnp.float32)
        if flip is None:
            return stock
        key = random.fold_in(random.PRNGKey(DEEPSEA_MAP_KEY),
                             jnp.asarray(flip, jnp.int32))
        drawn = random.bernoulli(key, 0.5, (self.size, self.size))
        return jnp.where(flip > 0.5, drawn.astype(jnp.float32), stock)

    def reset(self, key, params):
        base, flip = FlipEnv._split(params)
        obs, state = self.env.reset(key, base)
        state = state.replace(action_mapping=self._action_map(flip))
        return obs.reshape(-1), state

    def step(self, key, state, action, params):
        base, flip = FlipEnv._split(params)
        obs, state, reward, done, info = self.env.step(key, state, action, base)
        # gymnax auto-resets on `done` through reset_env, which restores the
        # stock map; put the sub-task's back so the fixed-length scan keeps
        # running under the same map after an early termination.
        state = state.replace(action_mapping=self._action_map(flip))
        return obs.reshape(-1), state, reward, done, info

    def observation_space(self, params=None):
        from gymnax.environments import spaces
        return spaces.Box(0.0, 1.0, (self.size * self.size,), jnp.float32)

    def __getattr__(self, item):        # action_space, default_params, ...
        return getattr(self.env, item)


def make_gymnax_env(env_name, **kwargs):
    """``gymnax.make`` plus this module's wrappers: the one constructor.

    Every trainer and post-hoc pass builds the environment through this, so a
    ``DeepSea<N>-bsuite`` name gets the same wrapped env everywhere; every
    other name is ``gymnax.make(env_name)`` exactly as before.
    """
    import gymnax
    n = deepsea_size(env_name)
    if n is None:
        return gymnax.make(env_name, **kwargs)
    env, params = gymnax.make("DeepSea-bsuite", size=n, **kwargs)
    params = params.replace(unscaled_move_cost=0.0, max_steps_in_episode=n)
    return DeepSeaEnv(env), params


def wrap_actions(env):
    """The environment for ``task_type actions``: ``FlipEnv`` reverses the
    action order; ``DeepSeaEnv`` carries its own action sub-task (the map)
    and reads the same flag, so it is returned as is."""
    return env if isinstance(env, DeepSeaEnv) else FlipEnv(env)


def gymnax_task_type(cfg):
    """The gymnax trainers' family name for a run from EITHER trainer family.

    The gymnax trainers record ``task_type`` (noise | param | actions); the
    shared runners record ``task`` (the suite's ``describe()``) under physics
    and actions and nothing under obs_noise. One name, so the post-hoc passes
    key their jitted rollouts -- and wrap FlipEnv -- the same way for both.
    """
    if 'task_type' in cfg:
        return cfg['task_type']
    mod = (cfg.get('task') or {}).get('task_mod')
    return {'physics': 'param', 'actions': 'actions'}.get(mod, 'noise')


def saved_task_rows(cfg, ckpt_files, stock_params, env_name=None):
    """``(offsets, bodies, mults)`` per SAVED PHASE, for either trainer family.

    ``offsets`` is what is added to the observation, ``(T, obs_dim)``;
    ``bodies`` one env params pytree per phase (the body and action order the
    phase was trained under); ``mults`` the physics multiplier per phase (1.0
    where the family has none). ``T`` is the number of rows in the
    checkpoint's ``noise_vectors``, i.e. the phases that were saved.

    The gymnax trainers' runs (``task_type`` in the config) keep the offset in
    ``noise_vectors`` and the body in ``param_mults`` / ``action_flips``; the
    shared runners' runs keep ONE row per phase whose meaning the run's
    ``task`` block gives -- the multiplier itself under physics, the flip flag
    plus its cue under actions, the offset under obs_noise -- and their
    ``num_tasks`` is the number of DISTINCT sub-tasks, not the phase count. The
    post-hoc passes used to read the first convention only, and on a
    shared-runner physics run added the multiplier to the observation and
    counted ten bodies for twenty phases (2026-09-13, the PBT arm).
    """
    env_name = env_name or cfg['env']
    rows = np.asarray(ckpt_files['noise_vectors'], dtype=np.float32)
    if 'task_type' in cfg:
        bodies = _legacy_task_params(cfg, ckpt_files, stock_params, env_name)
        mults = (np.asarray(ckpt_files['param_mults'], dtype=np.float64)
                 if 'param_mults' in ckpt_files else np.ones(len(rows)))
        return rows, bodies, mults[:len(rows)]
    if not cfg.get('task'):
        # obs_noise on the shared runner: the row is the offset, the body the stock one.
        return rows, [stock_params] * len(rows), np.ones(len(rows))
    from source.envs.registry import make_env_for_run
    _env, spec, obs_dim, _action_dim = make_env_for_run(cfg)
    offsets, bodies, mults = [], [], []
    for row in rows:
        off = spec.obs_offset(row)
        offsets.append(np.zeros(obs_dim, dtype=np.float32) if np.ndim(off) == 0
                       else np.asarray(off, dtype=np.float32))
        bodies.append(spec.env_params(row))
        mults.append(float(row[0]) if spec.task_mod == 'physics' else 1.0)
    return np.stack(offsets), bodies, np.asarray(mults, dtype=np.float64)


def saved_task_params(cfg, ckpt_files, stock_params, env_name=None):
    """The env params each saved phase of a run was trained under; see
    ``saved_task_rows``, which also gives the offsets."""
    if 'task_type' not in cfg:
        return saved_task_rows(cfg, ckpt_files, stock_params, env_name)[1]
    return _legacy_task_params(cfg, ckpt_files, stock_params, env_name)


def _legacy_task_params(cfg, ckpt_files, stock_params, env_name=None):
    """The env params each sub-task of a SAVED run was trained under.

    ``cfg`` is the run's results.json and ``ckpt_files`` the dict-like
    checkpoints.npz beside it. Returns one params pytree per sub-task, in
    order, so a post-hoc pass can roll a saved agent under exactly the body
    and action order its sub-task had. The observation offset is NOT in here:
    it lives in ``noise_vectors`` and is added to the observation by the
    caller, as in training.

    One function for the three families (CLAUDE.md rule (a)). Before this
    existed, ``evaluate_continual.py`` resolved the family itself and
    ``behavioural_divergence.py`` resolved nothing, so under the `actions`
    and `param` families the latter rolled every sub-task on the stock body
    with the stock action order, and its F and BD columns described an
    environment no sub-task after the first was trained on.
    """
    env_name = env_name or cfg['env']
    # Absent -- every run made before 2026-09-08 -- it is the offset family.
    task_type = cfg.get('task_type', 'noise')
    num_tasks = int(cfg['num_tasks'])
    if task_type == 'param':
        if 'param_mults' not in ckpt_files:
            raise ValueError(f"{cfg.get('method')}/{env_name}: task_type=param "
                             f"but no param_mults saved")
        mults = np.asarray(ckpt_files['param_mults'], dtype=float)
        return [apply_physics(env_name, stock_params, cfg['param_name'],
                              float(mults[t]))
                for t in range(min(num_tasks, len(mults)))]
    if task_type == 'actions':
        # `action_flips` is the only record of the sequence, so a run
        # without it cannot be scored at all.
        if 'action_flips' not in ckpt_files:
            raise ValueError(f"{cfg.get('method')}/{env_name}: "
                             f"task_type=actions but no action_flips saved")
        flips = np.asarray(ckpt_files['action_flips'])
        return [FlippedParams(stock_params, jnp.float32(flips[t]))
                for t in range(min(num_tasks, len(flips)))]
    return [stock_params] * num_tasks


class TaskSpec:
    """What a sub-task vector means for one built gymnax environment.

    The gymnax counterpart of ``tasks_mjx.TaskSpec``: it sits in the
    ``env_params`` slot of every scoring function, and each of them resolves
    a sub-task row through it into ``(observation offset, EnvParams)`` --
    ``(row, base params)`` under obs_noise and ``(0, rescaled params)`` under
    physics. A bare ``EnvParams`` in that slot is accepted everywhere and means
    obs_noise, which is how every run made before this class existed, and the
    analysis scripts that rebuild an environment themselves, are bit-unchanged.
    """

    def __init__(self, env_name, base_params, task_mod='obs_noise',
                 physics=None, actions=None):
        if task_mod not in TASK_MODS:
            raise ValueError(f"task_mod must be one of {TASK_MODS}, got "
                             f"{task_mod!r}")
        self.env_name = env_name
        self.base_params = base_params
        self.task_mod = task_mod
        self.physics = dict(physics or {})
        self.actions = {'flips': [0, 1], 'cue': 0, **dict(actions or {})}
        self.actions['flips'] = [int(f) for f in np.atleast_1d(
            self.actions['flips'])]
        self.actions['cue'] = int(self.actions['cue'])
        if task_mod == 'physics':
            param = self.physics.get('param')
            if param not in PHYSICS_PARAMS[env_name]:
                raise ValueError(
                    f"{env_name} has no physics parameter {param!r}; have "
                    f"{sorted(PHYSICS_PARAMS[env_name])}")
            if self.physics.get('mult_range') is not None:
                lo, hi = self.physics['mult_range']
                self.physics['mult_range'] = [float(lo), float(hi)]
            if self.physics.get('mults'):
                self.physics['mults'] = [float(m) for m in self.physics['mults']]
            elif self.physics.get('mult_range') is None:
                raise ValueError('a physics TaskSpec needs `mults` or `mult_range`')

    @property
    def dim(self):
        """Width of one sub-task row: obs_dim for offsets, 1 for a multiplier,
        1 + obs_dim for a flip flag plus its (possibly zero) cue offset."""
        return 1 if self.task_mod == 'physics' else None

    def obs_offset(self, task):
        """What is added to the observation before the policy reads it."""
        if self.task_mod == 'obs_noise':
            return task
        if self.task_mod == 'actions':
            return jnp.reshape(task, (-1,))[1:]
        return 0.0

    def env_params(self, task):
        """The params the rollout runs under for sub-task ``task``."""
        if self.task_mod == 'physics':
            mult = jnp.reshape(task, (-1,))[0]
            return apply_physics(self.env_name, self.base_params,
                                 self.physics['param'], mult)
        if self.task_mod == 'actions':
            return FlippedParams(self.base_params, jnp.reshape(task, (-1,))[0])
        return self.base_params

    def describe(self):
        """For the run config: what the sub-task vectors were."""
        options = {}
        if self.task_mod == 'physics':
            options['physics'] = dict(self.physics)
        if self.task_mod == 'actions':
            options['actions'] = dict(self.actions)
        return {'task_mod': self.task_mod, 'options': options}


def _resolve(env_params, task):
    """``(offset, EnvParams)`` for one sub-task row, through a TaskSpec or not.

    The one place the two constructions meet. A bare EnvParams is the
    obs_noise path exactly as it always was -- the row is the offset and the
    params are the params -- so nothing on disk moves.
    """
    if isinstance(env_params, TaskSpec):
        return env_params.obs_offset(task), env_params.env_params(task)
    return task, env_params


def build_env(env_name, episode_length, task_options=None):
    """The environment plus what its sub-tasks are, built once for the run.

    Returns ``(env, env_params)`` where ``env_params`` is gymnax's own
    ``EnvParams`` under obs_noise -- the default, and bit-for-bit what every
    caller received before ``task_options`` existed -- or a ``TaskSpec``
    wrapping it under physics. ``task_options`` (``--task_options KEY=VALUE``)
    takes ``task_mod``, and for physics ``physics_param`` (a name from
    PHYSICS_PARAMS) and EITHER ``physics_mults``, the per-sub-task multipliers
    as a comma-separated list (``physics_mults=1.0,2.0``), OR
    ``physics_mult_range`` (``0.5,2.0``, or ``default`` for the environment's
    entry in ``GYMNAX_PHYSICS_TASKS``), the ICLR family's per-trial log-uniform
    draw -- see ``physics_multipliers``; for actions
    ``actions_flips`` (``0,1``: sub-task i's action order is reversed when
    its entry is 1) and ``actions_cue`` (1 adds section C's observation
    offset to the reversed sub-task, see above).
    """
    cfg = ENV_CONFIGS[env_name]
    options = dict(task_options or {})
    task_mod = options.pop('task_mod', cfg.get('task_mod', 'obs_noise'))
    env, env_params = make_gymnax_env(env_name)
    physics = dict(cfg.get('physics', {}))
    if 'physics_param' in options:
        physics['param'] = options.pop('physics_param')
    env_params = env_params.replace(max_steps_in_episode=episode_length)
    if 'physics_mults' in options:
        raw = options.pop('physics_mults')
        physics['mults'] = [float(x) for x in
                            (raw.split(',') if isinstance(raw, str)
                             else np.atleast_1d(raw))]
    if 'physics_mult_range' in options:
        raw = options.pop('physics_mult_range')
        if isinstance(raw, str) and raw.lower() == 'default':
            table = GYMNAX_PHYSICS_TASKS[env_name]
            physics['param'] = physics.get('param') or table['param']
            physics['mult_range'] = list(table['mult_range'])
        else:
            physics['mult_range'] = [float(x) for x in
                                     (raw.split(',') if isinstance(raw, str)
                                      else np.atleast_1d(raw))]
        # A range is a draw; a fixed list would contradict it.
        physics.pop('mults', None)
    actions = {}
    if 'actions_flips' in options:
        raw = options.pop('actions_flips')
        actions['flips'] = [int(float(x)) for x in
                            (raw.split(',') if isinstance(raw, str)
                             else np.atleast_1d(raw))]
    if 'actions_cue' in options:
        actions['cue'] = int(options.pop('actions_cue'))
    if options:
        raise ValueError(f'unknown task option(s) {sorted(options)}')
    if task_mod == 'obs_noise':
        return env, env_params
    if task_mod == 'actions':
        env = wrap_actions(env)
    return env, TaskSpec(env_name, env_params, task_mod, physics, actions)


def physics_multipliers(spec, num_tasks, first_task_clean=True, trial=1):
    """The sub-task multipliers for a run, as ``(num_tasks, 1)``.

    Two constructions, and which one a spec carries is recorded in its
    ``describe()``:

    ``mults``        sub-task i multiplies the parameter by ``mults[i]``,
                     cycling past the end -- the ant's ``friction_cycle``
                     convention with the values written out -- so every trial
                     faces the same pair and differs in its seed only. With
                     ``first_task_clean`` False the first entry is skipped.
    ``mult_range``   the ICLR physics family's draw: sub-task 0 is the stock
                     body and every later one is log-uniform over the range,
                     seeded from the TRIAL alone by ``physics_mult_sequence``
                     -- the very function the gymnax trainers call, so a run
                     here and a run there at the same trial face the same
                     bodies in the same order, and the `switch` schedule over
                     these `num_tasks` is exactly their `--task_period` cycle.
    """
    if spec.physics.get('mult_range') is not None:
        seq = physics_mult_sequence(trial, num_tasks + (0 if first_task_clean else 1),
                                    tuple(spec.physics['mult_range']))
        if not first_task_clean:
            seq = seq[1:]
        return jnp.asarray(seq, dtype=jnp.float32)[:, None]
    mults = [float(m) for m in spec.physics['mults']]
    if not first_task_clean and len(mults) > 1:
        mults = mults[1:]
    seq = [mults[i % len(mults)] for i in range(num_tasks)]
    return jnp.asarray(seq, dtype=jnp.float32)[:, None]


def task_vectors(env_params, trial, num_tasks, obs_dim, noise_range,
                 first_task_clean=True):
    """The sub-task sequence for a trial, as ``(num_tasks, D)``.

    The observation-offset draw, ``task_noise_vectors``, unless ``env_params``
    is a physics ``TaskSpec``, where it is the multiplier cycle. This is what
    ``suites.Suite.task_vectors`` calls; a bare EnvParams goes down the first
    branch with the same arguments as before, bit-unchanged.
    """
    if isinstance(env_params, TaskSpec) and env_params.task_mod == 'physics':
        return physics_multipliers(env_params, num_tasks, first_task_clean,
                                   trial=trial)
    if isinstance(env_params, TaskSpec) and env_params.task_mod == 'actions':
        flips = env_params.actions['flips']
        if 'DeepSea' in str(env_params.env_name):
            # DeepSea's sub-task is a MAP: one per distinct sub-task, folded
            # with the trial -- the flags the gymnax trainers draw. Cycling
            # the reversal default [0, 1] gave the shared runner (PBT-PPO)
            # two maps, stock and map 1, instead of the NE / PPO arms' ten.
            flips = action_flip_sequence(num_tasks, 0, env_params.env_name,
                                         trial)
        flip = jnp.asarray([[float(flips[i % len(flips)])]
                            for i in range(num_tasks)], dtype=jnp.float32)
        if env_params.actions['cue']:
            # Section C's draw, sub-task 0 clean: the cue is the same offset
            # a `2task_n01`-style run at this trial would carry.
            offset = task_noise_vectors(trial, num_tasks, obs_dim,
                                        noise_range, True)
        else:
            offset = jnp.zeros((num_tasks, obs_dim), dtype=jnp.float32)
        return jnp.concatenate([flip, offset], axis=1)
    return task_noise_vectors(trial, num_tasks, obs_dim, noise_range,
                              first_task_clean)


def get_threshold(env, threshold_set=DEFAULT_THRESHOLD_SET):
    """Solved threshold for ``env`` under a named set. Raises on an unknown set."""
    if threshold_set not in THRESHOLD_SETS:
        raise KeyError(f"unknown threshold set {threshold_set!r}; "
                       f"have {sorted(THRESHOLD_SETS)}")
    return THRESHOLD_SETS[threshold_set][env]


def task_noise_vectors(trial, num_tasks, obs_dim, noise_range,
                       first_task_clean=True):
    """The sub-task sequence for a trial, as a ``(num_tasks, obs_dim)`` array.

    `_task_noise_vector_list` holds the draw itself and returns a list; this
    stacks it, which is the only difference and is what the vmapped scoring
    functions below want.

    ``first_task_clean`` keeps sub-task 0 unperturbed, which is the earlier
    study's convention and what makes the sequences nest. Setting it False
    draws every sub-task, giving two perturbed tasks that are symmetric with
    respect to the unperturbed environment -- closer to the two rugged
    landscapes of Wang & Dai, at the cost of the nesting property.
    """
    return jnp.stack(_task_noise_vector_list(
        trial, num_tasks, obs_dim, noise_range,
        first_task_clean=first_task_clean))


def obs_offset(env_params, task):
    """What the RL trainer adds to the observation for sub-task ``task``.

    The row itself under obs_noise -- a gymnax sub-task there IS an offset --
    and 0 under physics, where the sub-task is in the dynamics instead.
    """
    return _resolve(env_params, task)[0]


def rl_env_fns(env, env_params, num_envs):
    """``(reset(key, task), step(key, state, action, task))`` for PPO.

    The vectorised environment interface ``train_ppo`` drives; the mjx module
    has one of the same shape. Under obs_noise ``task`` changes nothing here
    -- the trainer adds the offset where the policy reads the observation --
    and the arithmetic is exactly what the trainer did inline before this
    existed, so its runs on disk are reproduced bit for bit. Under physics the
    environment steps with the sub-task's rescaled params.
    """
    def reset(key, task):
        params = _resolve(env_params, task)[1]
        obs, state = jax.vmap(lambda k: env.reset(k, params))(
            random.split(key, num_envs))
        return obs, state

    def step(key, state, action, task):
        params = _resolve(env_params, task)[1]
        next_obs, next_state, reward, done, _ = jax.vmap(
            lambda k, s, a: env.step(k, s, a, params))(
            random.split(key, num_envs), state, action)
        return next_obs, next_state, reward, done

    return reset, step


def make_scoring_fn(env, env_params, policy, param_template, episode_length,
                    num_evals):
    """Return ``score(genomes, key, noise_vector) -> (pop,)`` mean return.

    ``genomes`` is ``(pop, num_params)``. Every individual is run for
    ``num_evals`` episodes on independent reset keys and the returns averaged,
    so a genome's score is its expected return under the environment's own
    randomness rather than under one lucky reset.
    """
    episode_fn = make_episode_fn(env, policy, param_template, episode_length)

    def score(genomes, key, noise_vector):
        offset, params = _resolve(env_params, noise_vector)
        pop = genomes.shape[0]
        keys = random.split(key, pop * num_evals)
        repeated = jnp.repeat(genomes, num_evals, axis=0)
        returns = jax.vmap(episode_fn, in_axes=(0, 0, None, None))(
            repeated, keys, offset, params)
        return returns.reshape(pop, num_evals).mean(axis=1)

    return score


def make_fixed_seed_scoring_fn(env, env_params, policy, param_template,
                               episode_length, num_evals, eval_seed=0):
    """Like ``make_scoring_fn`` but with the *same* reset keys every call.

    For the landscape: two nearby points must differ because the genomes differ,
    not because they drew different resets. Common random numbers turn the
    return into a deterministic function of the genome, which is what a contour
    plot of it is implicitly claiming. Never use this for training -- a fixed
    evaluation set is exactly the thing an evolutionary search will overfit.
    """
    episode_fn = make_episode_fn(env, policy, param_template, episode_length)
    eval_keys = random.split(random.key(eval_seed), num_evals)

    def score(genomes, noise_vector):
        offset, params = _resolve(env_params, noise_vector)
        per_eval = jax.vmap(
            lambda k: jax.vmap(episode_fn, in_axes=(0, None, None, None))(
                genomes, k, offset, params))(eval_keys)
        return per_eval.mean(axis=0)

    return score


# ============================================================================
# Behaviour descriptors, for Dominated Novelty Search
# ============================================================================

def descriptor_dim(env_name):
    """Width of the hand-crafted descriptor for an environment."""
    return 2  # every entry in `handcrafted_descriptor` below returns two numbers


def handcrafted_descriptor(last_obs, env_name):
    """Where the episode ended, as two numbers.

    Taken from `source/algorithms/ne/dns.py handcrafted_descriptors`:
    the last valid observation, cut down to the two coordinates that describe
    what the policy *did* rather than how well it scored.

        CartPole      cart position, pole angle
        MountainCar   position, velocity
        Acrobot       cos and sin of the first joint

    "Last valid" matters. gymnax auto-resets on termination and the rollout
    scan runs the full episode length regardless, so the observation at the
    final step of the scan is often a fresh reset state and describes nothing.
    `make_descriptor_episode_fn` therefore freezes the observation at the first
    termination, which is what this receives.
    """
    if 'CartPole' in env_name:
        return jnp.stack([last_obs[0], last_obs[2]])
    if 'MountainCar' in env_name:
        return last_obs[:2]
    if 'Acrobot' in env_name:
        return last_obs[:2]
    n = deepsea_size(env_name)
    if n is not None:                   # row, column of the final cell
        idx = jnp.argmax(last_obs)
        return jnp.stack([idx // n, idx % n]).astype(jnp.float32)
    return last_obs[:2]


def make_trajectory_scoring_fn(env, env_params, policy, param_template,
                               episode_length, num_evals, traj_steps=50):
    """``score(genomes, key, noise) -> (fitness, observations)``.

    The AURORA path. Where `make_descriptor_scoring_fn` reduces an episode to
    two hand-picked numbers, this carries the whole (sub-sampled) observation
    trajectory, ``(pop, traj_steps, obs_dim)``, and the descriptor is whatever
    the learned encoder makes of it. That is the reference's default
    (`--descriptor aurora`) and the one its published gymnax numbers used.

    Sub-sampling follows `episode_relative_indices` from the reference rather
    than spreading samples over the episode CAP. The difference is not small on
    these environments: a 67-step Acrobot episode sampled evenly over 0..499
    gives two real states and forty-eight copies of the frozen final one, so
    the trajectory would describe termination instead of behaviour. Spreading
    the same number of samples over the steps that actually happened keeps
    every sample inside the episode.

    Only the FIRST evaluation's trajectory is kept when ``num_evals > 1``,
    matching the reference: the fitness is averaged over evaluations but the
    descriptor is one representative rollout, since averaging trajectories from
    different resets would describe no episode that happened.
    """
    num_traj_steps = min(traj_steps, episode_length)

    def episode(flat_params, key, noise_vector, env_params):
        params = unflatten_params(flat_params, param_template)
        obs, state = env.reset(key, env_params)

        def step_fn(carry, _):
            obs, state, total, done_flag, key = carry
            logits = policy.apply(params, obs + noise_vector)
            action = jnp.argmax(logits)
            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(
                step_key, state, action, env_params)
            total = total + reward * (1.0 - done_flag)
            still_running = 1.0 - done_flag
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
            return ((next_obs, next_state, total, done_flag, key),
                    (obs, still_running))

        (_, _, total, _, _), (all_obs, valid) = jax.lax.scan(
            step_fn, (obs, state, 0.0, 0.0, key), None, length=episode_length)
        return total, all_obs[episode_relative_indices(valid, num_traj_steps)]

    def score(genomes, key, noise_vector):
        offset, params = _resolve(env_params, noise_vector)
        pop = genomes.shape[0]
        keys = random.split(key, pop * num_evals)
        repeated = jnp.repeat(genomes, num_evals, axis=0)
        returns, trajectories = jax.vmap(episode, in_axes=(0, 0, None, None))(
            repeated, keys, offset, params)
        trajectories = trajectories.reshape(pop, num_evals, num_traj_steps, -1)
        return (returns.reshape(pop, num_evals).mean(axis=1),
                trajectories[:, 0])

    return score


def make_descriptor_scoring_fn(env, env_params, policy, param_template,
                               episode_length, num_evals, env_name):
    """``score(genomes, key, noise) -> (fitness, descriptors)``.

    Same rollout and same return as `make_scoring_fn`; it additionally carries
    the observation the episode ended on. Used only by DNS, which needs a
    behaviour descriptor to select on -- every other method pays nothing for it.
    """

    def episode(flat_params, key, noise_vector, env_params):
        params = unflatten_params(flat_params, param_template)
        obs, state = env.reset(key, env_params)

        def step_fn(carry, _):
            obs, state, total, done_flag, frozen, key = carry
            logits = policy.apply(params, obs + noise_vector)
            action = jnp.argmax(logits)
            key, step_key = random.split(key)
            next_obs, next_state, reward, done, _ = env.step(
                step_key, state, action, env_params)
            total = total + reward * (1.0 - done_flag)
            # Freeze the observation at the first termination; afterwards the
            # env has auto-reset and its observations describe a new episode.
            frozen = jnp.where(done_flag > 0, frozen, next_obs)
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
            return (next_obs, next_state, total, done_flag, frozen, key), None

        (_, _, total, _, frozen, _), _ = jax.lax.scan(
            step_fn, (obs, state, 0.0, 0.0, obs, key), None,
            length=episode_length)
        return total, handcrafted_descriptor(frozen, env_name)

    def score(genomes, key, noise_vector):
        offset, params = _resolve(env_params, noise_vector)
        pop = genomes.shape[0]
        keys = random.split(key, pop * num_evals)
        repeated = jnp.repeat(genomes, num_evals, axis=0)
        returns, descs = jax.vmap(episode, in_axes=(0, 0, None, None))(
            repeated, keys, offset, params)
        return (returns.reshape(pop, num_evals).mean(axis=1),
                descs.reshape(pop, num_evals, -1).mean(axis=1))

    return score
