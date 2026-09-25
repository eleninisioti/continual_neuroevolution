"""Rebuild a finished run's environment and policy, and roll its saved agents.

The post-hoc passes -- zero-shot transfer (`source/studies/evaluate_continual.py`),
forgetting and behavioural divergence (`scripts/analysis/behavioural_divergence.py`)
and the per-unit dormancy checkpoints (`scripts/analysis/plasticity_checkpoints.py`)
-- all start from the same question: given a `results.json` and a
`checkpoints.npz`, what environment was this run in, what network was it
searching, and what does agent `t` do on sub-task `s`? On gymnax each pass
answered it for itself, with `gymnax.make` and the classic-control MLP written
into the pass. That is why none of them could read a MiniGrid run.

This is the one answer, for every suite the shared runners drive (MiniGrid
now; brax and kinetix once their modules supply the three functions below).
It goes through `source/envs/registry.make_env_for_run`, so a run's own
recorded `task` block decides what a row of `noise_vectors` means -- an
environment index here, a friction multiplier on the ant -- and the pass never
has to know.

What a suite module must supply beyond the training interface:

    make_trace_fn(env, env_params, policy, param_template, episode_length)
        -> trace(flat_params, key, task) -> (states, occupancy, alive, ret)
        One episode with argmax actions: the states the policy saw, in
        whatever compact form the suite likes; a per-step occupancy feature
        (a position, an observation); the pre-termination mask; the return.
    policy_input(states)      compact states -> the batch the policy consumes
    random_policy_observations(env, env_params, task, num_obs, seed)
        -> (num_obs, obs_dim) policy inputs under a uniform random policy on
        sub-task `task`, the probe batch for dormancy. A random policy belongs
        to no arm, which is what keeps a dormancy number from being confounded
        with how good the policy is.

The gymnax passes are NOT routed through here. Their sub-tasks are
observation offsets, rescaled bodies and reversed action orders, resolved by
`saved_task_params` into per-sub-task env params; that path predates this
file and is left alone.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from source.algorithms.networks import unflatten_params
from source.envs.registry import get_suite, make_env_for_run, suite_for


def run_config(blob: dict) -> dict:
    """The run's settings from a loaded `results.json`, whichever trainer wrote it.

    The gymnax trainers put the interesting fields at the top level and nest
    the argparse namespace under `config`; the shared runners write only the
    nested block (plus summary fields). The top level wins where both have a
    key -- that is the one every arm agrees on.
    """
    return {**(blob.get('config') or {}),
            **{k: v for k, v in blob.items() if k != 'config'}}


def is_gymnax_run(cfg: dict) -> bool:
    return suite_for(cfg['env']) == 'gymnax'


class RunContext:
    """Environment, policy and jitted rollouts for one run configuration.

    Built once per distinct `(env, task block, hidden_dims, arch,
    episode_length, episodes)` and shared across runs, because the jit
    compilation dominates the cost of a single run.

    `task` arguments everywhere are ROWS of the run's `noise_vectors`.
    """

    def __init__(self, cfg: dict, episodes: int):
        self.suite = get_suite(suite_for(cfg['env']))
        self.env_name = cfg['env']
        self.episode_length = int(cfg['episode_length'])
        self.episodes = int(episodes)
        env, env_params, obs_dim, action_dim = make_env_for_run(cfg)
        self.env, self.env_params = env, env_params
        self.obs_dim, self.num_actions = int(obs_dim), int(action_dim)
        arch = dict(cfg.get('arch') or {})
        self.policy, self.template, self.num_params = self.suite.build_policy(
            random.key(0), obs_dim, action_dim, tuple(cfg['hidden_dims']), **arch)
        # The hidden activation ReDo's criterion is calibrated to. The head
        # decides it for the RL arms (`actors.head_for`), and the NE policy on
        # a suite is the same network, so it is the suite's.
        from source.studies.generalists.actors import head_for
        head = head_for(self.suite.name, self.suite.action_dims(env))
        self.activation = head.policy_activation
        # A Gaussian head means `policy.apply` returns an ACTION in [-1, 1]
        # rather than logits: the passes compare actions by distance
        # (`normalized_action_distance`) instead of by argmax, and the
        # softmax-based diagnostics do not apply.
        self.continuous = head.name == 'gaussian'
        # The conv policy reports its own hidden layers; an MLP is walked by
        # parameter name (`redo.hidden_activations`).
        self.hidden_activations_fn = getattr(self.policy, 'hidden_activations',
                                             None)

        trace = self.suite.make_trace_fn(env, env_params, self.policy,
                                         self.template, self.episode_length)
        episodes_of = jax.vmap(trace, in_axes=(None, 0, None))

        def _roll_on_task(agents, key, task):
            keys = random.split(key, self.episodes)
            return jax.vmap(episodes_of, in_axes=(0, None, None))(agents, keys, task)

        @jax.jit
        def roll_on_task(agents, key, task):
            """Every agent on one sub-task, on the SAME episode seeds, so a
            difference between two rows is a difference between policies."""
            return _roll_on_task(agents, key, task)

        @jax.jit
        def roll_cross(agents, keys, tasks):
            """Every agent on EVERY sub-task in one call: `(T, A, E, ...)`
            occupancy, alive mask and returns, sub-task t rolled under
            `keys[t]`, exactly what `roll_on_task` returns for each t in turn.
            The visited states are left out: for 20 x 20 x 20 x 1000 steps
            they are half a gigabyte nobody reads on this path, and XLA drops
            the computation with the output. One call rather than T because
            an MJX step is launch-bound (see `returns_own_tasks`)."""
            def on_task(key, task):
                _states, occ, alive, ret = _roll_on_task(agents, key, task)
                return occ, alive, ret
            return jax.vmap(on_task)(keys, tasks)

        @jax.jit
        def roll_own_tasks(agents, key, tasks):
            """Agent i on sub-task i, for every i."""
            keys = random.split(key, self.episodes)
            return jax.vmap(episodes_of, in_axes=(0, None, 0))(agents, keys, tasks)

        @jax.jit
        def returns_of(agent, key, task):
            """`(episodes,)` per-episode returns of one agent on one sub-task."""
            keys = random.split(key, self.episodes)
            return jax.vmap(trace, in_axes=(None, 0, None))(agent, keys, task)[3]

        @jax.jit
        def returns_own_tasks(agents, key, tasks):
            """`(A, episodes)` returns of agent i on sub-task i, in ONE call.

            The returns alone, not the trace: `roll_own_tasks` hands back
            every visited state too, which for 20 agents x 100 episodes x
            1000 MJX steps is a 136 MB array nobody asked for. One call for
            the whole checkpoint is what makes the evaluator's cost per run
            a handful of launches rather than one per (agent, sub-task): an
            MJX step is launch-bound, so 2000 episodes cost what 100 do.
            """
            keys = random.split(key, self.episodes)

            def one(agent, task):
                return jax.vmap(trace, in_axes=(None, 0, None))(agent, keys, task)[3]
            return jax.vmap(one)(agents, tasks)

        @jax.jit
        def logits_of(agents, states):
            """(A, P) agents on N compact states -> (A, N, num_actions)."""
            obs = self.suite.policy_input(states)

            def one(flat):
                return self.policy.apply(unflatten_params(flat, self.template), obs)
            return jax.vmap(one)(agents)

        self.roll_on_task = roll_on_task
        self.roll_cross = roll_cross
        self.roll_own_tasks = roll_own_tasks
        self.returns_of = returns_of
        self.returns_own_tasks = returns_own_tasks
        self.logits_of = logits_of
        self._probes = {}

    def probe(self, task, num_obs, seed=0):
        """The random-policy probe batch for sub-task `task`, cached by value."""
        key = (tuple(np.asarray(task, dtype=np.float64).reshape(-1).round(9)),
               int(num_obs), int(seed))
        if key not in self._probes:
            self._probes[key] = self.suite.random_policy_observations(
                self.env, self.env_params, jnp.asarray(task), num_obs, seed)
        return self._probes[key]

    @staticmethod
    def cache_key(cfg: dict, episodes: int):
        # `obs_norm`: a whitened NE run and an RL run of the same shape need
        # different traces (see `source/envs/mjx.py`, the post-hoc section).
        return (cfg['env'], repr(cfg.get('task')), tuple(cfg['hidden_dims']),
                repr(sorted((cfg.get('arch') or {}).items())),
                int(cfg['episode_length']), int(episodes),
                bool(cfg.get('obs_norm')),
                # Recorded whitening statistics: two runs that recorded
                # different ones must not share a rebuilt environment.
                tuple(cfg.get('obs_mean') or ()), tuple(cfg.get('obs_std') or ()))


def live_states(states, alive):
    """The states actually visited, flattened over episodes and steps.

    `states` is `(E, T, ...)` and `alive` `(E, T)`; the result keeps each
    state's own trailing shape, so a suite's compact form survives.
    """
    states = np.asarray(states)
    flat = states.reshape((-1,) + states.shape[2:])
    return flat[np.asarray(alive).reshape(-1) > 0]
