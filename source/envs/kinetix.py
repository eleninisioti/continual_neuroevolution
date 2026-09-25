"""The Kinetix task family for this study: the sub-task is WHICH LEVEL.

On Kinetix (Matthews et al. 2025), the JAX 2D-physics benchmark, over its
twenty hand-designed MEDIUM levels -- `m/h0_unicycle` .. `m/h19_thrust_left_
very_easy`. A sub-task is one of those levels, which makes this the same shape
as the MiniGrid suite (a sub-task is one of two environments) rather than
gymnax's (a sub-task is an offset added to one environment's observation).

## This reproduces a configuration that worked, and says where it does not

The previous codebase's `source/studies/kinetix/ga_continual.py` found a
setting under which the GA solved all twenty levels in the continual chain
(`projects/kinetix/budget_g200_r3_gafinal`). Everything the environment side
of that setting decides is reproduced here EXACTLY, and the parameter count is
the assertion: 125x125x3 pixels, multi-discrete actions, frame skip 2, an
episode of 256 steps, and `ActorOnlyPixelsRNN` at `recurrent_model: false` --
1,128,256 parameters, checked at build time in `build_policy`.

Two things are deliberately NOT reproduced, and both are repo-wide conventions
that the old kinetix trainers were the only holdout from:

* **the reported action is the argmax**, where the old trainer sampled from
  the policy during evolution AND in its end-of-level verify. Every other body
  here reports a deterministic policy, and the working run's own verify says
  the difference is small on a solved level (16/16 solved, mean +1.112, spread
  +1.107..+1.132, against +1.108 in training). The SEARCH is unaffected: NE
  selects on whatever `make_scoring_fn` returns, and that is the argmax return
  for every method equally.
* **the network is ours, not Kinetix's object.** `KinetixPixelsPolicy` in
  `source/algorithms/networks.py` is `ActorOnlyPixelsRNN` with the dead carry
  and the (T, B) axes removed -- see its docstring. Same layers, same widths,
  same initialisers, same parameter count. What this buys is that
  `policy.apply(params, obs)` works, which is the one call every scoring
  function, PPO rollout, churn probe and dormancy probe in this repo makes.

## Why the levels can be stacked instead of switched

All twenty medium levels share IDENTICAL `StaticEnvParams` and `EnvParams` --
checked, not assumed: they differ only in `EnvState`, the layout, whose pytree
has the same structure and the same leaf shapes on all twenty. So the whole
level set is one stacked `EnvState` and a sub-task is an INDEX into it. There
is no `lax.switch` here and no per-level compile: one environment object, one
rollout, one compilation, and switching levels inside a jitted loop costs a
gather.

That is why this suite is cheap where the MiniGrid one had to switch once per
episode: two xminigrid environments are two different objects, twenty Kinetix
levels are twenty rows of one array.

Auto-reset goes back to the SAME level because `KinetixEnv.step` takes an
`override_reset_state`, and this suite always hands it the current level's
initial state. Nothing is drawn from `train_levels_list`, so the vendored
`kinetix_config_pixels.yaml` is not read at all: the level JSON's own
`EnvParams` is what the working runs used (the old trainer overwrote the
config with it immediately after loading), and it is what runs here.

## The action space

Multi-discrete, as the working configuration: six independent categoricals --
four motor bindings with three choices (reverse / off / forward) and two
thruster bindings with two (off / on) -- emitted as one flat vector of 16
logits. `action_dim` is therefore the LOGIT count, 16, and `action_dims` is
the per-dimension choice count, `(3, 3, 3, 3, 2, 2)`; PPO's head is built
from the latter (`source/studies/generalists/actors.py:multi_discrete_head`).

## Descriptors

`kinetix.util.behaviour` already holds them, written for the old DNS trainers
and reused unchanged:

    duty_factor          (6,)  the hand-designed descriptor -- the fraction
                               of the episode each actuator binding is active.
                               It is the one descriptor that means the same
                               thing on all twenty levels, which a
                               coordinate-based one does not.
    step_features        (13,) the AURORA input: green position and velocity,
                               blue position, their distance, and the six
                               action channels.

Thirteen, not the 46,876-dim pixel observation: an LSTM auto-encoder over full
frames would cost more than the search it serves and would mostly encode the
renderer. `traj_feature_dim` is the suite hook that tells `run_nes` so.

## Solved

`solved_threshold` is 1.0: a Kinetix episode return crosses 1.0 exactly when
the goal was reached (the terminal bonus), which is the criterion the old
trainers' verify used (`ga_continual.py`: `_vr >= 1.0`). Unlike the mjx
bodies, this body HAS a threshold, so the breadth columns are available.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from source.algorithms.networks import (
    create_kinetix_pixels_policy_network,
    create_kinetix_transformer_policy_network,
    unflatten_params,
)
from source.envs.kinetix_levels import CELL_ALL, CELLS, LEVELS
from source.metrics.aurora import episode_relative_indices

__all__ = [
    'ENV_CONFIGS', 'LEVELS', 'CELLS', 'TaskSpec', 'LevelSet', 'build_env',
    'env_dims', 'action_dims', 'task_vectors', 'obs_offset', 'build_policy',
    'make_scoring_fn', 'make_fixed_seed_scoring_fn',
    'make_descriptor_scoring_fn', 'make_trajectory_scoring_fn',
    'descriptor_dim', 'traj_feature_dim', 'rl_env_fns',
    'make_trace_fn', 'policy_input', 'random_policy_observations',
]


# ---------------------------------------------------------------------------
# The level set
# ---------------------------------------------------------------------------

# The observation and the action space are properties of the medium env size,
# which every one of the twenty levels shares. Written down so `env_dims` and
# `build_policy` agree with `check()` without building an environment.
IMAGE_SHAPE = (125, 125, 3)             # 500x500 screen at downscale 4
GLOBAL_INFO_DIM = 1                     # [gravity_y / 10]
OBS_DIM = int(np.prod(IMAGE_SHAPE)) + GLOBAL_INFO_DIM      # 46,876
NUM_MOTOR_BINDINGS = 4
NUM_THRUSTER_BINDINGS = 2
ACTION_DIMS = (3,) * NUM_MOTOR_BINDINGS + (2,) * NUM_THRUSTER_BINDINGS
ACTION_DIM = int(sum(ACTION_DIMS))      # 16 logits
PARAM_COUNT = 1_128_256                 # the working GA configuration's

# The SECOND observation, since 2026-09-23: Kinetix's `symbolic_entity`, which
# its transformer network reads (`configs/env/entity.yaml` +
# `configs/model/model-transformer.yaml`). One row per circle / polygon /
# directed joint / thruster, their masks, a (4, 9, 9) attention mask and the
# joint/thruster -> shape indexes, at the medium env size (3 circles, 6
# polygons, 2 joints seen from both ends, 2 thrusters). Flattened in THIS
# order, which `KinetixTransformerPolicy` reads back; `build_env` checks every
# shape against a real reset.
ENTITY_LAYOUT = (
    ('circles', (3, 19)), ('polygons', (6, 27)), ('joints', (4, 22)),
    ('thrusters', (2, 8)), ('circle_mask', (3,)), ('polygon_mask', (6,)),
    ('joint_mask', (4,)), ('thruster_mask', (2,)),
    ('attention_mask', (4, 9, 9)), ('joint_indexes', (4, 2)),
    ('thruster_indexes', (2,)),
)
ENTITY_OBS_DIM = int(sum(np.prod(shape) for _, shape in ENTITY_LAYOUT))  # 672
OBSERVATIONS = ('pixels', 'entity')

# The AURORA input width and the hand-designed descriptor width, from
# `kinetix.util.behaviour`. Constants rather than imports so this module can be
# read (and `check()` run) without pulling in jax2d.
TRAJ_FEATURE_DIM = 7 + NUM_MOTOR_BINDINGS + NUM_THRUSTER_BINDINGS   # 13
DESCRIPTOR_DIM = NUM_MOTOR_BINDINGS + NUM_THRUSTER_BINDINGS         # 6


# ---------------------------------------------------------------------------
# The cells
# ---------------------------------------------------------------------------
#
# A cell is what `--env` names and what a directory in the run tree is named
# after. `Kinetix20` is the continual chain over all twenty levels;
# `Kinetix-<level>` is the stationary run on one of them -- twenty of those,
# and they are what the non-continual block runs.
#
# Every cell builds the SAME environment object and the SAME policy: the level
# set a cell carries only decides which rows of the stacked initial state the
# schedule can reach. So a stationary run and the continual chain differ in the
# schedule and in nothing else.

_BASE = {
    # 128 steps of frame-skip 2. The working GA configuration ran 256, and
    # nothing happens after 128: every solving incumbent scores the same at
    # 96/128/160/192/256 and the GA trained at 128 solves the same 20 of 20
    # levels (source/studies/kinetix/settings.py, 2026-09-13). Not a cap the
    # reward depends on (unlike MiniGrid's), just the scan length; a solved
    # level ends in 19-150 steps. `settings.check()` asserts the two agree.
    'episode_length': 128,
    # POLICY_ARCH['kinetix'] -- fc_layer_depth 5 x fc_layer_width 128, tanh.
    'hidden_dims': (128,) * 5,
    'task_mod': 'level',
    # A sub-task is a level index, not a perturbation; there is no sigma here.
    'noise_range': 0.0,
    # A return crosses 1.0 exactly when the goal was reached.
    'solved_threshold': 1.0,
    # The population is scored in chunks of this many genomes. 512 x 3 pixel
    # episodes at once is tens of GB of activations; the old trainer chunked
    # at 32 for the same reason and this is that number. It changes nothing
    # about what is computed -- `lax.scan` over chunks, then reshape.
    'eval_batch_size': 32,
}

# LEVELS, CELL_ALL and CELLS come from `source/envs/kinetix_levels.py`, which
# imports nothing -- see that module for why the split exists.

ENV_CONFIGS = {name: {**_BASE, 'levels': levels}
               for name, levels in CELLS.items()}


class TaskSpec:
    """What a sub-task vector means here: the index of a level.

    A row of ``noise_vectors`` is one number, the index into this cell's level
    tuple as a float. ``describe()`` records the level list so a finished run's
    environment can be rebuilt from its config (``registry.make_env_for_run``).
    """

    def __init__(self, levels, observation='pixels'):
        self.levels = tuple(levels)
        self.observation = observation
        self.task_mod = 'level'

    @property
    def dim(self):
        return 1

    def obs_offset(self, task):
        return 0.0

    def describe(self):
        options = {'levels': list(self.levels)}
        # Recorded only off the default, so every pixel run's config is what
        # it always was; `registry.make_env_for_run` hands it back to
        # `build_env`, which is how a post-hoc pass rebuilds the right body.
        if self.observation != 'pixels':
            options['observation'] = self.observation
        return {'task_mod': 'level', 'options': options}


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------

def _flat_obs(obs):
    """``PixelsObservation`` -> the flat vector every consumer here takes."""
    return jnp.concatenate([obs.image.reshape(-1), obs.global_info])


def _flat_entity_obs(obs):
    """``EntityObservation`` -> one float vector in ``ENTITY_LAYOUT`` order.
    Masks and indexes are small exact integers, so float32 carries them."""
    return jnp.concatenate([
        jnp.asarray(getattr(obs, name), jnp.float32).reshape(-1)
        for name, _ in ENTITY_LAYOUT])


class LevelSet:
    """Kinetix's twenty levels as ONE environment plus a stacked initial state.

    ``reset(level, key) -> (obs, state)`` and
    ``step(level, state, action, key) -> (obs, state, reward, done)``, where
    ``level`` is a traced int32 index into ``self.levels``. Both are single
    environment calls -- there is no switch and no per-level compile, because
    every level shares this environment's static and env params (asserted in
    ``build_env``) and differs only in its initial ``EnvState``.

    ``initial(level)`` is the gather that makes that true, and it is also what
    ``step`` hands ``override_reset_state``, so Kinetix's own auto-reset
    returns to the level the rollout is on rather than to a sampled one.
    """

    def __init__(self, levels, env, env_params, init_states, episode_length,
                 observation='pixels'):
        self.levels = tuple(levels)
        self.env = env
        self.env_params = env_params
        # One pytree whose every leaf has a leading axis of len(levels).
        self.init_states = init_states
        self.episode_length = int(episode_length)
        self.observation = observation
        self._flatten = _flat_entity_obs if observation == 'entity' else _flat_obs
        self.obs_dim = ENTITY_OBS_DIM if observation == 'entity' else OBS_DIM
        self.action_dim = ACTION_DIM
        self.action_dims = ACTION_DIMS

    def initial(self, level):
        return jax.tree.map(lambda x: x[level], self.init_states)

    def reset(self, level, key):
        obs, state = self.env.reset(key, env_params=self.env_params,
                                    override_reset_state=self.initial(level))
        return self._flatten(obs), state

    def step(self, level, state, action, key):
        obs, state, reward, done, _info = self.env.step(
            key, state=state, action=action, env_params=self.env_params,
            # Kinetix auto-resets on `done`; this is what sends it back to the
            # SAME level. Without it there is no reset function at all and the
            # step raises.
            override_reset_state=self.initial(level))
        return self._flatten(obs), state, reward, done


def build_env(env_name, episode_length, task_options=None):
    """``(env, spec)``: the level set behind one step, and what a sub-task is.

    ``task_options`` takes ``levels`` (a comma-separated string, or the list a
    recorded config carries) and ``observation``: ``pixels`` (the default, and
    every run before 2026-09-23) or ``entity``, Kinetix's symbolic-entity
    observation that its transformer reads. The observation decides the
    network (`build_policy`), as in Kinetix's own `make_network_from_config`.
    """
    from kinetix.environment.env import make_kinetix_env
    from kinetix.environment.utils import ActionType, ObservationType
    from kinetix.util.saving import load_from_json_file

    cfg = ENV_CONFIGS[env_name]
    options = dict(task_options or {})
    options.pop('task_mod', None)
    levels = options.pop('levels', cfg['levels'])
    if isinstance(levels, str):
        levels = [x.strip() for x in levels.split(',') if x.strip()]
    levels = tuple(levels)
    observation = options.pop('observation', 'pixels')
    if observation not in OBSERVATIONS:
        raise ValueError(f'unknown Kinetix observation {observation!r}; '
                         f'have {OBSERVATIONS}')
    unknown = [x for x in levels if x not in LEVELS]
    if unknown:
        raise ValueError(f'unknown Kinetix level(s) {unknown}; have {LEVELS}')
    if not levels:
        raise ValueError('a Kinetix run needs at least one level')
    if options:
        raise ValueError(f'unknown task option(s) {sorted(options)}')

    loaded = [load_from_json_file(f'm/{name}') for name in levels]
    states = [x[0] for x in loaded]
    static_params = [x[1] for x in loaded]
    env_params = [x[2] for x in loaded]

    # The assumption the whole design rests on, checked rather than trusted: a
    # level that brought its own static or env params would need a switch, and
    # stacking it here would silently run it under level 0's physics.
    from flax.serialization import to_state_dict
    ref_static = to_state_dict(static_params[0])
    ref_params = to_state_dict(env_params[0])
    for name, sp, ep in zip(levels[1:], static_params[1:], env_params[1:]):
        if to_state_dict(sp) != ref_static:
            raise ValueError(f'level {name!r} has different StaticEnvParams '
                             f'from {levels[0]!r}; this suite stacks levels '
                             'and cannot mix env sizes')
        cur = to_state_dict(ep)
        if any(not np.array_equal(np.asarray(cur[k]), np.asarray(ref_params[k]))
               for k in ref_params):
            raise ValueError(f'level {name!r} has different EnvParams from '
                             f'{levels[0]!r}; this suite stacks levels')

    static = static_params[0]
    if (int(static.num_motor_bindings) != NUM_MOTOR_BINDINGS
            or int(static.num_thruster_bindings) != NUM_THRUSTER_BINDINGS):
        raise ValueError(
            f'expected {NUM_MOTOR_BINDINGS} motor and '
            f'{NUM_THRUSTER_BINDINGS} thruster bindings, got '
            f'{int(static.num_motor_bindings)} and '
            f'{int(static.num_thruster_bindings)}')

    # `reset_fn=None`: this suite never samples a level, it always overrides
    # the reset state with the one it wants, so there is nothing for a reset
    # function to do and `train_levels_list` plays no part.
    env = make_kinetix_env(
        action_type=ActionType.MULTI_DISCRETE,
        observation_type=(ObservationType.SYMBOLIC_ENTITY
                          if observation == 'entity'
                          else ObservationType.PIXELS),
        reset_fn=None,
        env_params=env_params[0],
        static_env_params=static,
    )
    if observation == 'entity':
        # The layout the policy unflattens is a constant; check it against
        # what the renderer actually emits rather than trust it.
        obs, _ = env.reset(jax.random.key(0), env_params=env_params[0],
                           override_reset_state=states[0])
        got = tuple((name, tuple(getattr(obs, name).shape))
                    for name, _ in ENTITY_LAYOUT)
        if got != ENTITY_LAYOUT:
            raise ValueError(f'Kinetix entity observation is {got}, not the '
                             f'ENTITY_LAYOUT {ENTITY_LAYOUT}')
    init_states = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    if len(levels) == 1:
        # `tree.map(f, x)` with one tree does not stack; give it the axis.
        init_states = jax.tree.map(lambda x: x[None], states[0])
    return (LevelSet(levels, env, env_params[0], init_states, episode_length,
                     observation),
            TaskSpec(levels, observation))


def env_dims(env):
    return int(env.obs_dim), int(env.action_dim)


def action_dims(env):
    """The per-dimension choice count PPO's multi-discrete head is built from."""
    return env.action_dims


def task_vectors(spec, trial, num_tasks, obs_dim, noise_range,
                 first_task_clean=True):
    """Sub-task i is level ``i % len(levels)``, as a ``(num_tasks, 1)`` array.

    Every trial faces the same levels in the same order; a trial is a seed.
    The other arguments are the shared signature and unused here.
    """
    del trial, obs_dim, noise_range, first_task_clean
    seq = [i % len(spec.levels) for i in range(num_tasks)]
    return jnp.asarray(seq, dtype=jnp.float32)[:, None]


def obs_offset(spec, task):
    return 0.0


def _level_index(task):
    return jnp.reshape(task, (-1,))[0].astype(jnp.int32)


def build_policy(key, obs_dim, action_dim, hidden_dims):
    """``(policy, param_template, num_params)``: the actor-only network.

    The OBSERVATION decides the network, as it does in Kinetix: the flat pixel
    vector (46,876) gets the conv policy, the flat entity vector (672) gets
    Kinetix's transformer (`KinetixTransformerPolicy`). The two sizes cannot
    collide, and anything else raises.

    The pixel parameter count is asserted, not reported: it is the one number
    that says this is still the network the working GA configuration evolved.
    """
    if int(obs_dim) == ENTITY_OBS_DIM:
        policy, param_template = create_kinetix_transformer_policy_network(
            key, ENTITY_LAYOUT, int(action_dim), tuple(hidden_dims))
        num_params = int(
            jax.flatten_util.ravel_pytree(param_template)[0].shape[0])
        return policy, param_template, num_params
    if int(obs_dim) != OBS_DIM:
        raise ValueError(f'expected obs_dim {OBS_DIM} (pixels) or '
                         f'{ENTITY_OBS_DIM} (entity), got {obs_dim}')
    policy, param_template = create_kinetix_pixels_policy_network(
        key, IMAGE_SHAPE, GLOBAL_INFO_DIM, int(action_dim),
        tuple(hidden_dims))
    num_params = int(jax.flatten_util.ravel_pytree(param_template)[0].shape[0])
    if num_params != PARAM_COUNT:
        raise ValueError(
            f'Kinetix policy has {num_params} parameters, not the '
            f'{PARAM_COUNT} of the configuration this reproduces -- the '
            'architecture has drifted from ActorOnlyPixelsRNN')
    return policy, param_template, num_params


# ---------------------------------------------------------------------------
# Rollouts and scoring
# ---------------------------------------------------------------------------

def _rollout(env, policy, param_template, episode_length, collect=False):
    """``episode(flat_params, key, task) -> return``, or with the trace.

    One undiscounted episode return with ARGMAX actions -- per action
    dimension, since the action is six independent categoricals. Reward accrues
    until the first termination and the scan always runs the full length, so it
    vmaps; this is the contract every other suite's rollout has.

    ``collect`` also returns the per-step 13-value ``step_features`` (the
    AURORA input and the raw material for the duty-factor descriptor), the
    raw action indices, and a validity mask. The pixel frames are NOT kept:
    they are 46,875 floats a step, so one scoring call's worth is hundreds of
    GB, and no descriptor here reads them.
    """
    from kinetix.util.behaviour import step_features

    offsets = jnp.asarray(np.cumsum((0,) + ACTION_DIMS[:-1]), dtype=jnp.int32)
    width = int(max(ACTION_DIMS))
    valid_choice = jnp.asarray(
        np.arange(width)[None, :] < np.asarray(ACTION_DIMS)[:, None], bool)

    def greedy_action(logits):
        """Per-dimension argmax over the padded (6, 3) logit block."""
        cols = jnp.arange(width)
        idx = jnp.clip(offsets[:, None] + cols[None, :], 0, ACTION_DIM - 1)
        block = jnp.where(valid_choice, jnp.take(logits, idx), -jnp.inf)
        return jnp.argmax(block, axis=-1)

    def episode(flat_params, key, task):
        level = _level_index(task)
        params = unflatten_params(flat_params, param_template)
        reset_key, scan_key = random.split(key)
        obs, state = env.reset(level, reset_key)

        def step_fn(carry, _):
            obs, state, total, done_flag, key = carry
            key, step_key = random.split(key)
            action = greedy_action(policy.apply(params, obs))
            feat = step_features(state, action, NUM_MOTOR_BINDINGS)
            next_obs, next_state, reward, done = env.step(
                level, state, action, step_key)
            total = total + reward * (1.0 - done_flag)
            valid = 1.0 - done_flag
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
            return ((next_obs, next_state, total, done_flag, key),
                    ((feat, action, valid) if collect else None))

        (_, _, total, _, _), trace = jax.lax.scan(
            step_fn, (obs, state, 0.0, 0.0, scan_key), None,
            length=episode_length)
        if not collect:
            return total
        feats, actions, valid = trace
        return total, feats, actions, valid

    return episode


def _chunked(fn, genomes, chunk):
    """``fn`` over the population in chunks of ``chunk``, results concatenated.

    A 512 x 3 batch of 125x125x3 rollouts does not fit; this is the same
    `lax.scan`-over-chunks the previous trainer used (`eval_batch_size 32`)
    and it changes nothing about what is computed. `chunk` must divide the
    population, which the settings module asserts.
    """
    pop = genomes.shape[0]
    if pop % chunk:
        raise ValueError(f'population {pop} is not divisible by the Kinetix '
                         f'eval_batch_size {chunk}')
    batched = genomes.reshape((pop // chunk, chunk) + genomes.shape[1:])
    _, out = jax.lax.scan(lambda c, b: (c, fn(b)), None, batched)
    return jax.tree.map(lambda x: x.reshape((pop,) + x.shape[2:]), out)


def make_scoring_fn(env, env_params, policy, param_template, episode_length,
                    num_evals):
    """``score(genomes, key, task) -> (pop,)`` mean return over ``num_evals``."""
    del env_params
    episode = _rollout(env, policy, param_template, episode_length)
    chunk = _BASE['eval_batch_size']

    def score(genomes, key, task):
        eval_keys = random.split(key, num_evals)

        def one_chunk(batch):
            per_eval = jax.vmap(
                lambda k: jax.vmap(episode, in_axes=(0, None, None))(
                    batch, k, task))(eval_keys)
            return per_eval.mean(axis=0)

        return _chunked(one_chunk, genomes, min(chunk, genomes.shape[0]))

    return score


def make_fixed_seed_scoring_fn(env, env_params, policy, param_template,
                               episode_length, num_evals, eval_seed=0):
    """Same keys every call, for the landscapes. Never for training."""
    del env_params
    episode = _rollout(env, policy, param_template, episode_length)
    eval_keys = random.split(random.key(eval_seed), num_evals)
    chunk = _BASE['eval_batch_size']

    def score(genomes, task):
        def one_chunk(batch):
            per_eval = jax.vmap(
                lambda k: jax.vmap(episode, in_axes=(0, None, None))(
                    batch, k, task))(eval_keys)
            return per_eval.mean(axis=0)

        return _chunked(one_chunk, genomes, min(chunk, genomes.shape[0]))

    return score


# ============================================================================
# The post-hoc passes (`source/envs/run_context.RunContext`)
# ============================================================================
#
# What `evaluate_continual`, `behavioural_divergence` and
# `plasticity_checkpoints` need beyond the training interface. The pixel frame
# is what makes this body different from the other suites here: 46,876 floats
# a step, so a trace cannot keep what the policy saw. It keeps the 13-value
# `step_features` instead -- enough for the return, the state-visitation
# feature and the episode mask, which is all the ZT pass reads.

def make_trace_fn(env, env_params, policy, param_template, episode_length):
    """``trace(flat_params, key, task) -> (states, occupancy, alive, ret)``.

    ``states`` are the per-step ``step_features`` (``(T, 13)``: green xy and
    velocity, blue xy, their distance, the six action channels) -- NOT the
    frames, see above. ``occupancy`` is the green body's xy, the natural
    state-visitation feature on a physics level. ``alive`` masks the steps
    before the first termination.
    """
    del env_params
    episode = _rollout(env, policy, param_template, episode_length,
                       collect=True)

    def trace(flat_params, key, task):
        total, feats, _actions, valid = episode(flat_params, key, task)
        return feats, feats[:, :2], valid, total

    return trace


def policy_input(states):
    """Compact states -> the batch the policy consumes.

    Not available on this body: a compact state here is `step_features`, and
    the policy consumes the rendered frame, which the trace does not keep (one
    behavioural-divergence sweep's worth of frames is tens of GB). Behavioural
    divergence on Kinetix needs a trace that keeps the physics `EnvState` and
    re-renders it here; until then the BD column is `--` for this body and the
    F column, which is returns only, is not affected.
    """
    raise NotImplementedError(
        'kinetix.policy_input: the trace keeps step_features, not frames; '
        'behavioural divergence needs an EnvState-keeping trace (see the '
        'docstring)')


def random_policy_observations(env, env_params, task, num_obs, seed=0,
                               steps_per_episode=32):
    """``(num_obs, obs_dim)`` flat pixel observations under a UNIFORM RANDOM
    multi-discrete policy on level ``task``, the dormancy probe batch. Many
    short episode prefixes rather than one long walk, as on MiniGrid: a random
    walk's frames are strongly autocorrelated and a probe wants coverage.
    512 x 46,876 floats is 96 MB, which is why the default probe size is kept.
    """
    del env_params
    level = _level_index(jnp.asarray(task))
    episodes = -(-num_obs // steps_per_episode)
    choices = jnp.asarray(ACTION_DIMS, dtype=jnp.float32)

    def one(key):
        reset_key, act_key, scan_key = random.split(key, 3)
        obs, state = env.reset(level, reset_key)
        u = random.uniform(act_key, (steps_per_episode, len(ACTION_DIMS)))
        actions = jnp.floor(u * choices).astype(jnp.int32)

        def step_fn(carry, action):
            obs, state, done_flag, key = carry
            key, step_key = random.split(key)
            next_obs, next_state, _r, done = env.step(level, state, action,
                                                      step_key)
            emitted = (obs, 1.0 - done_flag)
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
            return (next_obs, next_state, done_flag, key), emitted

        _, (obs_seq, alive) = jax.lax.scan(
            step_fn, (obs, state, 0.0, scan_key), actions)
        return obs_seq, alive

    obs_seq, alive = jax.jit(jax.vmap(one))(
        random.split(random.key(seed), episodes))
    obs_seq = np.asarray(obs_seq).reshape(-1, obs_seq.shape[-1])
    alive = np.asarray(alive).reshape(-1) > 0
    live = obs_seq[alive]
    if len(live) < num_obs:
        live = np.concatenate([live, obs_seq[~alive]])[:num_obs]
    return jnp.asarray(live[:num_obs])


# ============================================================================
# Behaviour descriptors, for Dominated Novelty Search
# ============================================================================

def descriptor_dim(env_name):
    del env_name
    return DESCRIPTOR_DIM


def traj_feature_dim(env_name):
    """The width of what ``make_trajectory_scoring_fn`` returns.

    13, the `step_features` vector -- NOT the 46,876-dim pixel observation the
    shared runner would otherwise hand AURORA. See the module docstring.
    """
    del env_name
    return TRAJ_FEATURE_DIM


def make_descriptor_scoring_fn(env, env_params, policy, param_template,
                               episode_length, num_evals, env_name):
    """``score(genomes, key, task) -> (fitness, descriptors)``.

    The descriptor is `kinetix.util.behaviour.duty_factor`: the fraction of the
    episode each of the six actuator bindings was driven. It is the only
    hand-designed descriptor here that means the same thing on every level.
    """
    del env_params, env_name
    from kinetix.util.behaviour import duty_factor
    episode = _rollout(env, policy, param_template, episode_length,
                       collect=True)
    chunk = _BASE['eval_batch_size']

    def one(flat_params, key, task):
        total, _feats, actions, valid = episode(flat_params, key, task)
        return total, duty_factor(actions, valid, NUM_MOTOR_BINDINGS,
                                  NUM_THRUSTER_BINDINGS)

    def score(genomes, key, task):
        eval_keys = random.split(key, num_evals)

        def one_chunk(batch):
            returns, descs = jax.vmap(
                lambda k: jax.vmap(one, in_axes=(0, None, None))(
                    batch, k, task))(eval_keys)
            return returns.mean(axis=0), descs.mean(axis=0)

        return _chunked(one_chunk, genomes, min(chunk, genomes.shape[0]))

    return score


def make_trajectory_scoring_fn(env, env_params, policy, param_template,
                               episode_length, num_evals, traj_steps=50):
    """``score(genomes, key, task) -> (fitness, trajectories)``, the AURORA path.

    Sub-sampled with ``episode_relative_indices`` as on the other suites, so
    the samples land inside the episode that happened. Only the first
    evaluation's trajectory is kept when ``num_evals > 1``, matching what DNS
    feeds the encoder elsewhere.
    """
    del env_params
    num_traj_steps = min(traj_steps, episode_length)
    episode = _rollout(env, policy, param_template, episode_length,
                       collect=True)
    chunk = _BASE['eval_batch_size']

    def one(flat_params, key, task):
        total, feats, _actions, valid = episode(flat_params, key, task)
        return total, feats[episode_relative_indices(valid, num_traj_steps)]

    def score(genomes, key, task):
        eval_keys = random.split(key, num_evals)

        def one_chunk(batch):
            returns, trajs = jax.vmap(
                lambda k: jax.vmap(one, in_axes=(0, None, None))(
                    batch, k, task))(eval_keys)
            return returns.mean(axis=0), trajs[0]

        return _chunked(one_chunk, genomes, min(chunk, genomes.shape[0]))

    return score


# ============================================================================
# PPO
# ============================================================================

def rl_env_fns(env, env_params, num_envs):
    """``(reset(key, task), step(key, state, action, task))`` for PPO.

    Vectorised over ``num_envs``; the level index is a scalar shared by the
    batch. Kinetix's own auto-reset does the resetting, back to the SAME level
    because `LevelSet.step` passes that level's initial state as
    ``override_reset_state`` -- so, unlike the MiniGrid suite, there is no
    reset-and-select here.
    """
    del env_params

    def reset(key, task):
        level = _level_index(task)
        return jax.vmap(lambda k: env.reset(level, k))(
            random.split(key, num_envs))

    def step(key, state, action, task):
        level = _level_index(task)
        return jax.vmap(lambda k, s, a: env.step(level, s, a, k))(
            random.split(key, num_envs), state, action)

    return reset, step
