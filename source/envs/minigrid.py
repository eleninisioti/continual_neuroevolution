"""The MiniGrid task family for this study: the sub-task is WHICH ENVIRONMENT.

On XLand-MiniGrid's pure-JAX ports of the MiniGrid environments (Nikulin et
al. 2023, ``xminigrid``), and the only suite whose two sub-tasks are two
different environments rather than two settings of one. Every MiniGrid
environment shares a SYMBOLIC 7x7 egocentric view and the six MiniGrid actions
(forward, right, left, pick up, drop, toggle), so nothing has to be padded:
one network plays any pair of them as it stands.

Why this body: a 512-sample search gradient on a pixel CNN is mostly noise, so
the NE arms could not learn the specialists on a pixel body at all -- which is
what a dropped MinAtar suite established before it was removed on 2026-09-08.
A MiniGrid view is 98 categorical values, the network below has ~8k
parameters, and a random policy already reaches the goal in a quarter of
Empty-8x8 episodes, so the question "does switching leave one agent good at
both?" can be asked where the NE arms plausibly learn the specialists first.

## The observation

xminigrid's view is ``(7, 7, 2)`` of uint8: a tile id (13 kinds) and a colour
id (12) per cell. Ids are categorical, so the view is one-hot encoded into
``13 + 12 = 25`` binary planes, ``(7, 7, 25)``, and carried FLAT (1225
values), because every consumer of an observation in this repo takes a
vector. The policy is the conv net of Young & Tian 2019
(``GridConvPolicy``: one 3x3 valid convolution, one Dense, the head),
reshaping the vector back to planes on the way in, at conv 4 / dense 64 --
7,758 parameters. As in the original MiniGrid the agent's own inventory is
NOT in the view: after picking a key up the key simply disappears from it.

## Episode caps and reward

xminigrid keeps MiniGrid's per-environment cap (``max_steps``: 256 on
Empty-8x8, 1024 on the 16x16 rooms, 640 on DoorKey-8x8) and its reward,
``1 - 0.9 * t / max_steps`` on reaching the goal and 0 otherwise, so a
return is in [0, 1] and shorter solutions score higher. The cap is part of
the reward, so this suite leaves each environment's own in place, and
``episode_length`` (ENV_CONFIGS: 1024) is only the scan length: it must
cover the longer of the pair's caps and ``build_env`` refuses a pair it does
not.

## Why the environment is a traced argument, and where the switch sits

This study switches sub-task inside one jitted loop, so the environment index
is a traced argument and a switching run compiles once. Rather than stepping
the active environment under ``lax.switch`` INSIDE the per-step scan, this
suite's NE rollout switches once per EPISODE: ``lax.switch`` on the index selects a whole single-
environment episode (its own reset and scan), so a generation costs one
environment's steps and the conditional runs once per generation rather
than once per step. That placement was forced on 2026-09-07: with XLA's
CUDA-graph command buffers on, a per-step conditional inside a 1024-step
scan crashed every switch-schedule run within minutes (segfaults and cuDNN
illegal-address faults); with them off (``XLA_FLAGS=
--xla_gpu_enable_command_buffer=``) the same runs sat at generation 0,
because a conditional executed outside a graph reads its predicate on the
host at every scan step. The RL step (``rl_env_fns``) still carries both
environments' timesteps and switches per step, as PPO's rollout is 50 steps
and ran at speed either way. ``scripts/outdated/generalists/check_minigrid_tasks.py``
asserts on the CPU that a switched rollout on environment i returns exactly
what xminigrid's own environment i returns under the same keys.

## No solved threshold

As on the mjx bodies: ``solved_threshold`` is None and the
analysis reports the generalist SCORE with the specialist rows as the scale.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from source.algorithms.networks import (
    create_grid_conv_policy_network,
    unflatten_params,
)
from source.metrics.aurora import episode_relative_indices

__all__ = [
    'ENV_CONFIGS', 'ENVS', 'MultiEnv', 'TaskSpec', 'build_env',
    'build_policy', 'descriptor_dim', 'make_scoring_fn',
    'make_fixed_seed_scoring_fn', 'make_descriptor_scoring_fn',
    'make_trajectory_scoring_fn', 'task_vectors', 'obs_offset', 'rl_env_fns',
    'make_trace_fn', 'policy_input', 'random_policy_observations',
]

# Short name -> xminigrid id. Every registered MiniGrid environment shares the
# 7x7 view and the six actions; these are the ones a feed-forward policy can
# plausibly play (the Memory ones need recurrence by construction).
ENVS = {
    'Empty-5x5': 'MiniGrid-Empty-5x5',
    'Empty-6x6': 'MiniGrid-Empty-6x6',
    'Empty-8x8': 'MiniGrid-Empty-8x8',
    'Empty-16x16': 'MiniGrid-Empty-16x16',
    'EmptyRandom-5x5': 'MiniGrid-EmptyRandom-5x5',
    'EmptyRandom-6x6': 'MiniGrid-EmptyRandom-6x6',
    'EmptyRandom-8x8': 'MiniGrid-EmptyRandom-8x8',
    'EmptyRandom-16x16': 'MiniGrid-EmptyRandom-16x16',
    'DoorKey-5x5': 'MiniGrid-DoorKey-5x5',
    'DoorKey-6x6': 'MiniGrid-DoorKey-6x6',
    'DoorKey-8x8': 'MiniGrid-DoorKey-8x8',
    'DoorKey-16x16': 'MiniGrid-DoorKey-16x16',
    'FourRooms': 'MiniGrid-FourRooms',
    'Unlock': 'MiniGrid-Unlock',
    'UnlockPickUp': 'MiniGrid-UnlockPickUp',
    'LockedRoom': 'MiniGrid-LockedRoom',
    'BlockedUnlockPickUp': 'MiniGrid-BlockedUnlockPickUp',
}

VIEW = (7, 7)            # xminigrid's default view_size, asserted in build_env
NUM_TILES = 13           # xminigrid.core.constants.Tiles, ids 0..12
NUM_COLORS = 12          # xminigrid.core.constants.Colors, ids 0..11
NUM_PLANES = NUM_TILES + NUM_COLORS
NUM_ACTIONS = 6          # forward, right, left, pick up, drop, toggle

# The default pair, settled 2026-09-07 from the probes below: the 8x8 room
# with a random start and the 16x16 one, both with the goal in the corner.
# A nested pair -- the 16x16 specialist solves both (0.95 / 0.97 at 32
# fixed seeds) and the 8x8 specialist reaches 0.73 on the 16x16 room -- so
# a generalist exists and NES finds it from the 16x16 side in 200
# generations. `envs=A,B` in --task_options chooses another pair; `ENVS`
# above is the menu, and these are the probes (NES sigma 0.1 / lr 0.05,
# 300 generations, one trial, the held-out return in [0, 1]):
#
#     EmptyRandom-8x8    0.9-1.0 by generation 200, every NES and GA cell
#     EmptyRandom-16x16  1.0 by generation 200 (NES 0.1/0.05 and 0.3/0.15)
#     FourRooms          0.1-0.2 at 300 (1000-generation run pending)
#     DoorKey-5x5/6x6/8x8, Unlock, UnlockPickUp, LockedRoom
#                        0, and provably stuck: of 1024 fresh argmax
#                        policies (sigma 0.1 or 0.5 around the init) NONE
#                        scores on any of them, so every NES utility is
#                        z-scored from a constant and GA drifts (0 after
#                        300 generations on DoorKey-5x5). A key-and-door
#                        task needs exploration the study's deterministic
#                        policies do not have.
#
# The network is the grid conv policy at conv 4 / dense 64;
# `arch` is what a trainer hands `build_policy` beyond hidden_dims. There
# is no observation offset on this suite, so `noise_range` is 0 and unused.
_MINIGRID = {
    'envs': ('EmptyRandom-8x8', 'EmptyRandom-16x16'),
    'hidden_dims': (64,),
    'arch': {'conv_features': 4},
    # The scan length; each environment keeps its own cap (module docstring).
    # 1024 is the 16x16 room's, the longest of the default pair.
    'episode_length': 1024,
    'task_mod': 'env',
    'noise_range': 0.0,
    'solved_threshold': None,
}

ENV_CONFIGS = {
    'MiniGrid': dict(_MINIGRID),
    # The name the 2026-09-07 probes on the 16x16 room ran under, before the
    # pair became the default; the same configuration.
    'MiniGrid-L1024': dict(_MINIGRID),
}


class TaskSpec:
    """What a sub-task vector means here: the index of an environment.

    A row of ``noise_vectors`` is one number, the environment index as a
    float, and ``describe()`` records the environment list so a finished
    run's environment can be rebuilt from its config
    (``suites.make_env_for_run``).
    """

    def __init__(self, envs):
        self.envs = tuple(envs)
        self.task_mod = 'env'

    @property
    def dim(self):
        return 1

    def obs_offset(self, task):
        return 0.0

    def describe(self):
        return {'task_mod': 'env', 'options': {'envs': list(self.envs)}}


def encode_view(view):
    """``(7, 7, 2)`` uint8 ids -> ``(7 * 7 * 25,)`` float32 one-hot planes."""
    tiles = jax.nn.one_hot(view[..., 0], NUM_TILES, dtype=jnp.float32)
    colors = jax.nn.one_hot(view[..., 1], NUM_COLORS, dtype=jnp.float32)
    return jnp.concatenate([tiles, colors], axis=-1).reshape(-1)


class MultiEnv:
    """Several xminigrid environments behind one reset/step, selected by index.

    ``reset(key, env) -> (obs, timesteps)`` and
    ``step(timesteps, action, env) -> (obs, timesteps, reward, done, pos)``,
    where ``timesteps`` is a tuple holding every environment's xminigrid
    ``TimeStep`` (only the active one moves), ``obs`` is the active one's
    encoded view, ``done`` its ``last()`` flag and ``pos`` the agent's
    position scaled to [0, 1] by that environment's grid, for the behaviour
    descriptor. ``env`` is a scalar int32, traced. xminigrid's step takes no
    key (the state carries its own); reset does. Both are single-environment
    functions; callers ``vmap`` them.
    """

    def __init__(self, envs, episode_length):
        import xminigrid
        self.names = tuple(envs)
        self.envs, self.params = [], []
        for name in self.names:
            env, params = xminigrid.make(ENVS[name])
            if tuple(env.observation_shape(params)[:2]) != VIEW:
                raise ValueError(f'{name}: view {env.observation_shape(params)}'
                                 f', expected {VIEW}')
            if env.num_actions(params) != NUM_ACTIONS:
                raise ValueError(f'{name} has {env.num_actions(params)} '
                                 f'actions, expected {NUM_ACTIONS}')
            if int(params.max_steps) > int(episode_length):
                raise ValueError(
                    f'{name} caps an episode at {params.max_steps} steps, '
                    f'longer than episode_length={episode_length}; raise it')
            self.envs.append(env)
            self.params.append(params)
        self.obs_dim = VIEW[0] * VIEW[1] * NUM_PLANES

    def _scale(self, i):
        p = self.params[i]
        return jnp.asarray([p.height - 1, p.width - 1], jnp.float32)

    # Single-environment access, for the NE rollout's per-episode switch. The
    # reset key is the i-th of `len(envs)` splits, exactly what `reset` hands
    # environment i, so an episode is the same draw either way.
    def reset_one(self, i, key):
        ts = self.envs[i].reset(self.params[i],
                                random.split(key, len(self.envs))[i])
        return encode_view(ts.observation), ts

    def step_one(self, i, ts, action):
        ts = self.envs[i].step(self.params[i], ts, action)
        pos = ts.state.agent.position.astype(jnp.float32) / self._scale(i)
        return (encode_view(ts.observation), ts,
                ts.reward.astype(jnp.float32), ts.last(), pos)

    def reset(self, key, env):
        keys = random.split(key, len(self.envs))
        steps = tuple(e.reset(p, k)
                      for e, p, k in zip(self.envs, self.params, keys))
        obs = jax.lax.switch(env, [(lambda t: (lambda _: encode_view(
            t.observation)))(t) for t in steps], None)
        return obs, steps

    def step(self, steps, action, env):
        def branch(i):
            def run(_):
                ts = self.envs[i].step(self.params[i], steps[i], action)
                new_steps = tuple(ts if j == i else steps[j]
                                  for j in range(len(steps)))
                pos = ts.state.agent.position.astype(jnp.float32) / self._scale(i)
                return (encode_view(ts.observation), new_steps,
                        ts.reward.astype(jnp.float32), ts.last(), pos)
            return run
        return jax.lax.switch(env, [branch(i) for i in range(len(self.envs))],
                              None)


def build_env(env_name, episode_length, task_options=None):
    """``(env, spec)``: the environments behind one step, and what a sub-task is.

    ``task_options`` takes ``envs`` (a comma-separated list, or a list from a
    recorded config).
    """
    cfg = ENV_CONFIGS[env_name]
    options = dict(task_options or {})
    options.pop('task_mod', None)
    envs = options.pop('envs', cfg['envs'])
    if isinstance(envs, str):
        envs = [e.strip() for e in envs.split(',') if e.strip()]
    envs = tuple(envs)
    unknown = [e for e in envs if e not in ENVS]
    if unknown:
        raise ValueError(f'unknown MiniGrid environment(s) {unknown}; have '
                         f'{sorted(ENVS)}')
    if len(envs) < 2:
        raise ValueError('a MiniGrid run needs at least two environments')
    if options:
        raise ValueError(f'unknown task option(s) {sorted(options)}')
    env = MultiEnv(envs, episode_length)
    return env, TaskSpec(envs)


def env_dims(env):
    return int(env.obs_dim), NUM_ACTIONS


def task_vectors(spec, trial, num_tasks, obs_dim, noise_range,
                 first_task_clean=True):
    """Sub-task i is environment ``i % len(envs)``, as a ``(num_tasks, 1)`` array.

    Every trial faces the same pair in the same order; a trial is a seed. The
    other arguments are the shared signature and unused here.
    """
    del trial, obs_dim, noise_range, first_task_clean
    seq = [i % len(spec.envs) for i in range(num_tasks)]
    return jnp.asarray(seq, dtype=jnp.float32)[:, None]


def obs_offset(spec, task):
    return 0.0


def _env_index(task):
    return jnp.reshape(task, (-1,))[0].astype(jnp.int32)


def build_policy(key, obs_dim, action_dim, hidden_dims, conv_features=4):
    """``(policy, param_template, num_params)``: the conv policy on the planes.

    ``conv_features`` comes from the environment's ``arch`` (ENV_CONFIGS); the
    trainers pass it through.
    """
    channels = int(obs_dim) // (VIEW[0] * VIEW[1])
    policy, param_template = create_grid_conv_policy_network(
        key, VIEW + (channels,), action_dim, tuple(hidden_dims),
        conv_features=int(conv_features))
    num_params = int(jax.flatten_util.ravel_pytree(param_template)[0].shape[0])
    return policy, param_template, num_params


def _rollout(env, policy, param_template, episode_length, collect=False):
    """``episode(flat_params, key, task) -> return``, or with the trace.

    One undiscounted episode return with argmax actions, the same contract as
    the other suites' rollouts: reward accrues until the first termination and
    the scan always runs the full length so it vmaps. ``collect`` also returns
    the per-step RAW views (``(T, 7, 7, 2)`` uint8, 98 bytes a step rather
    than the 4900 of the encoded planes -- 1536 episodes of 1024 steps would
    otherwise be 7.7 GB a scoring call), a validity mask and the agent's
    scaled position after each step. AURORA's consumer encodes the few
    steps it keeps.
    """

    def one_env(i):
        # A whole episode on environment i alone; the branches of the switch.
        def run(args):
            flat_params, key = args
            params = unflatten_params(flat_params, param_template)
            obs, ts = env.reset_one(i, key)

            def step_fn(carry, _):
                obs, ts, total, done_flag = carry
                action = jnp.argmax(policy.apply(params, obs))
                next_obs, next_ts, reward, done, pos = env.step_one(
                    i, ts, action)
                total = total + reward * (1.0 - done_flag)
                valid = 1.0 - done_flag
                done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
                return ((next_obs, next_ts, total, done_flag),
                        ((ts.observation, valid, pos) if collect else None))

            (_, _, total, _), trace = jax.lax.scan(
                step_fn, (obs, ts, 0.0, 0.0), None, length=episode_length)
            if not collect:
                return total
            all_obs, valid, positions = trace
            return total, all_obs, valid, positions
        return run

    branches = [one_env(i) for i in range(len(env.envs))]

    def episode(flat_params, key, task):
        return jax.lax.switch(_env_index(task), branches, (flat_params, key))

    return episode


def make_scoring_fn(env, env_params, policy, param_template, episode_length,
                    num_evals):
    """``score(genomes, key, task) -> (pop,)`` mean return over ``num_evals``."""
    del env_params
    episode = _rollout(env, policy, param_template, episode_length)

    def score(genomes, key, task):
        pop = genomes.shape[0]
        keys = random.split(key, pop * num_evals)
        repeated = jnp.repeat(genomes, num_evals, axis=0)
        returns = jax.vmap(episode, in_axes=(0, 0, None))(repeated, keys, task)
        return returns.reshape(pop, num_evals).mean(axis=1)

    return score


def make_fixed_seed_scoring_fn(env, env_params, policy, param_template,
                               episode_length, num_evals, eval_seed=0):
    """Same keys every call, for the landscapes. Never for training."""
    del env_params
    episode = _rollout(env, policy, param_template, episode_length)
    eval_keys = random.split(random.key(eval_seed), num_evals)

    def score(genomes, task):
        per_eval = jax.vmap(
            lambda k: jax.vmap(episode, in_axes=(0, None, None))(
                genomes, k, task))(eval_keys)
        return per_eval.mean(axis=0)

    return score


def rl_env_fns(env, env_params, num_envs):
    """``(reset(key, task), step(key, state, action, task))`` for PPO.

    Vectorised over ``num_envs``; the environment index is a scalar shared by
    the batch, so ``lax.switch`` runs one environment's step. A finished
    episode is reset on the step that finishes it and the reset observation
    handed back on that step -- gymnax's auto-reset, which is what the other
    suites' functions of the same name do. xminigrid's own step is
    deterministic given its state, so the key only feeds those resets.
    """
    del env_params

    def reset(key, task):
        index = _env_index(task)
        return jax.vmap(lambda k: env.reset(k, index))(
            random.split(key, num_envs))

    def step_one(key, steps, action, index):
        obs, new_steps, reward, done, _pos = env.step(steps, action, index)
        reset_obs, reset_steps = env.reset(key, index)
        obs = jnp.where(done, reset_obs, obs)
        new_steps = jax.tree.map(lambda r, n: jnp.where(done, r, n),
                                 reset_steps, new_steps)
        return obs, new_steps, reward, done

    def step(key, state, action, task):
        index = _env_index(task)
        return jax.vmap(lambda k, s, a: step_one(k, s, a, index))(
            random.split(key, num_envs), state, action)

    return reset, step


# ============================================================================
# Behaviour descriptors, for Dominated Novelty Search
# ============================================================================

def descriptor_dim(env_name):
    return 2


def handcrafted_descriptor(positions, valid):
    """Where the episode ended: the agent's scaled (row, column) at its last
    valid step, the classic maze descriptor. AURORA over the observation
    trajectory is the default, as on the other suites."""
    last = jnp.maximum(valid.sum().astype(jnp.int32) - 1, 0)
    return positions[last]


def make_descriptor_scoring_fn(env, env_params, policy, param_template,
                               episode_length, num_evals, env_name):
    """``score(genomes, key, task) -> (fitness, descriptors)``."""
    del env_params, env_name
    episode = _rollout(env, policy, param_template, episode_length,
                       collect=True)

    def one(flat_params, key, task):
        total, _all_obs, valid, positions = episode(flat_params, key, task)
        return total, handcrafted_descriptor(positions, valid)

    def score(genomes, key, task):
        pop = genomes.shape[0]
        keys = random.split(key, pop * num_evals)
        repeated = jnp.repeat(genomes, num_evals, axis=0)
        returns, descs = jax.vmap(one, in_axes=(0, 0, None))(
            repeated, keys, task)
        return (returns.reshape(pop, num_evals).mean(axis=1),
                descs.reshape(pop, num_evals, -1).mean(axis=1))

    return score


def make_trajectory_scoring_fn(env, env_params, policy, param_template,
                               episode_length, num_evals, traj_steps=50):
    """``score(genomes, key, task) -> (fitness, observations)``, the AURORA path.

    Sub-sampled with ``episode_relative_indices`` as on the other suites, so
    the samples land inside the episode that happened. Only the first
    evaluation's trajectory is kept when ``num_evals > 1``.
    """
    del env_params
    num_traj_steps = min(traj_steps, episode_length)
    episode = _rollout(env, policy, param_template, episode_length,
                       collect=True)

    def one(flat_params, key, task):
        total, views, valid, _positions = episode(flat_params, key, task)
        kept = views[episode_relative_indices(valid, num_traj_steps)]
        return total, jax.vmap(encode_view)(kept)

    def score(genomes, key, task):
        pop = genomes.shape[0]
        keys = random.split(key, pop * num_evals)
        repeated = jnp.repeat(genomes, num_evals, axis=0)
        returns, trajectories = jax.vmap(one, in_axes=(0, 0, None))(
            repeated, keys, task)
        trajectories = trajectories.reshape(pop, num_evals, num_traj_steps, -1)
        return (returns.reshape(pop, num_evals).mean(axis=1),
                trajectories[:, 0])

    return score


# ============================================================================
# Post-hoc measurement, for the passes that re-score SAVED agents
# ============================================================================
#
# `source/envs/run_context.py` builds a finished run's environment and policy
# back from its config and needs three things a training rollout does not
# expose: the states an agent visited (compactly), a cheap occupancy feature
# per step, and a probe batch under a policy that belongs to no arm. Each
# suite that the shared runners drive supplies them here; the gymnax passes
# have their own, older path and are untouched.

def make_trace_fn(env, env_params, policy, param_template, episode_length):
    """``trace(flat_params, key, task) -> (states, occupancy, alive, ret)``.

    ``states`` are the RAW ``(T, 7, 7, 2)`` uint8 views -- 98 bytes a step,
    against 4,900 for the encoded planes -- so a 20-agent x 20-episode sweep
    of 1024-step episodes is 40 MB rather than 2 GB; ``policy_input`` encodes
    whichever subset a pass keeps. ``occupancy`` is the agent's scaled
    ``(row, col)`` after each step, the natural state-visitation feature on a
    grid. ``alive`` masks the steps before the first termination.
    """
    del env_params
    episode = _rollout(env, policy, param_template, episode_length,
                       collect=True)

    def trace(flat_params, key, task):
        total, views, valid, positions = episode(flat_params, key, task)
        return views, positions, valid, total

    return trace


def policy_input(states):
    """``(N, 7, 7, 2)`` raw views -> ``(N, 7 * 7 * 25)`` planes, the policy's input."""
    return jax.vmap(encode_view)(jnp.asarray(states))


def random_policy_observations(env, env_params, task, num_obs, seed=0,
                               steps_per_episode=32):
    """``(num_obs, obs_dim)`` policy inputs under a UNIFORM RANDOM policy on
    sub-task ``task``, for the dormancy probes. Many short episode prefixes
    rather than one long walk, because a random walk's states are strongly
    autocorrelated and a probe wants coverage."""
    del env_params
    index = int(np.asarray(task).reshape(-1)[0])
    episodes = -(-num_obs // steps_per_episode)

    def one(key):
        reset_key, act_key = random.split(key)
        obs, ts = env.reset_one(index, reset_key)
        actions = random.randint(act_key, (steps_per_episode,), 0, NUM_ACTIONS)

        def step_fn(carry, action):
            obs, ts, done_flag = carry
            next_obs, next_ts, _r, done, _pos = env.step_one(index, ts, action)
            emitted = (obs, 1.0 - done_flag)
            done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
            return (next_obs, next_ts, done_flag), emitted

        _, (obs_seq, alive) = jax.lax.scan(step_fn, (obs, ts, 0.0), actions)
        return obs_seq, alive

    obs_seq, alive = jax.jit(jax.vmap(one))(
        random.split(random.key(seed), episodes))
    obs_seq = np.asarray(obs_seq).reshape(-1, obs_seq.shape[-1])
    alive = np.asarray(alive).reshape(-1) > 0
    live = obs_seq[alive]
    if len(live) < num_obs:
        # A random walk that terminates early on every episode; pad with the
        # (few) states there are rather than fail.
        live = np.concatenate([live, obs_seq[~alive]])[:num_obs]
    return jnp.asarray(live[:num_obs])
