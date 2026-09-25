"""One interface over the task families this study runs on.

``train_nes.run_nes`` was written against gymnax and touched the environment in
eight places -- make the env, read its obs/action dims, build the policy, draw
the sub-task offsets, build the training and evaluation scoring functions, and
build the two descriptor variants DNS needs. Everything else in that file --
the searchers, the schedule, the records, the artifacts -- never mentions an
environment at all, and neither does anything under
``scripts/outdated/generalists/analysis/`` that reads a table or draws a curve.

So a second task family is those eight things and nothing else. This module is
the interface, and a *suite* is the module that answers it:

    gymnax   ``source/envs/gymnax_classic.py``      CartPole / Acrobot / MountainCar
    mjx      ``source/envs/mjx.py``  CheetahRun / ant
    minigrid ``source/envs/minigrid.py`` MiniGrid (xminigrid),
             since 2026-09-07: the sub-task is WHICH ENVIRONMENT, on a
             symbolic 7x7 view where the NE arms can learn

A MinAtar suite sat between the two until 2026-09-08, where the sub-task was
which game. It was dropped: the experiments did not separate the methods, and
carrying a body no result rests on is a cost with no return. `GridConvPolicy`
in `source/algorithms/networks.py` outlives it -- MiniGrid runs on that network.

Since 2026-09-05 the interface has three more names, added for the ant and
for the RL arms on it: ``task_vectors`` (what a sub-task IS -- the mjx suite
makes the ant's a friction multiplier where everything else is an observation
offset), ``obs_offset`` (what an RL rollout adds to the observation, 0 when the
sub-task is a change to the physics) and ``rl_env_fns`` (a vectorised
reset/step pair for ``train_ppo``, so that trainer holds no branch on the
backend). All three have a default that reproduces the gymnax behaviour bit
for bit, so a suite that predates them needs no change. The gymnax suite
answers all three itself since the same date, because it too now has a
physics sub-task (``--task_options task_mod=physics``: a multiplier on the
pole length, the link masses or the engine force -- ``tasks.PHYSICS_PARAMS``),
and ``make_env_for_run`` rebuilds either kind from a finished run's config.

Both provide the same function names with the same signatures, which is why
this file is thin: it resolves ``env_name`` to a suite, holds the one genuine
difference between them (how an environment is constructed and how its
dimensions are read), and delegates the rest.

``ENV_NAMES`` is unique across suites, so ``suite_for('CheetahRun')`` needs no
flag anywhere -- ``--env`` alone selects the family, and the queue scripts,
``train_all.py`` and the analysis all keep taking one environment name.

## The key stream is unchanged

``make_env`` takes no key from the caller: the mjx backend needs a reset to
read its observation width, and it uses a fixed ``random.key(0)`` for it rather
than consuming from the run's stream. A gymnax run through this module is
therefore bit-identical to one through the old inline code, which is what
lets the runs already on disk stay comparable with anything run after it.
"""

from __future__ import annotations

class Suite:
    """A task family: its name, its module, and how to build one environment.

    Everything except `make_env` is delegated to the module, so a suite adds
    exactly one function to what its `tasks*.py` already exposes.
    """

    def __init__(self, name, module, make_env):
        self.name = name
        self.module = module
        # (env_name, episode_length, task_options=None)
        #     -> (env, env_params, obs_dim, action_dim)
        # `env_params` is whatever the suite needs to score a sub-task on that
        # env: gymnax's EnvParams, or the mjx suite's `TaskSpec`, which says
        # whether a sub-task vector is an observation offset or a friction
        # multiplier. `task_options` is the command line's `--task_options`,
        # ignored by gymnax.
        self.make_env = make_env

    def task_vectors(self, env_params, trial, num_tasks, obs_dim, noise_range,
                     first_task_clean=True):
        """The ``(num_tasks, D)`` sub-task array for a trial.

        A module that defines ``task_vectors`` decides D for itself (the mjx
        suite: obs_dim for offsets, 1 for a friction multiplier); one that
        does not gets the observation-offset draw every existing run used,
        called exactly as before, so gymnax is bit-unchanged.
        """
        fn = getattr(self.module, 'task_vectors', None)
        if fn is not None:
            return fn(env_params, trial, num_tasks, obs_dim, noise_range,
                      first_task_clean)
        return self.module.task_noise_vectors(trial, num_tasks, obs_dim,
                                              noise_range, first_task_clean)

    def obs_offset(self, env_params, task):
        """What the RL trainer adds to the observation for ``task``.

        The sub-task vector itself where a sub-task IS an observation offset
        -- gymnax, and the mjx bodies under obs_noise -- and 0 where it is a
        change to the physics instead.
        """
        fn = getattr(self.module, 'obs_offset', None)
        return task if fn is None else fn(env_params, task)

    def traj_feature_dim(self, env_name, obs_dim):
        """The width of a ``make_trajectory_scoring_fn`` row -- AURORA's input.

        The OBSERVATION width on every suite whose trajectories are
        observations, which is all of them except kinetix: a Kinetix
        observation is a 125x125x3 frame and an LSTM auto-encoder over full
        frames would cost more than the search it serves, so that suite
        returns a 13-value per-step feature vector instead and says so here.
        A suite that does not define it gets `obs_dim`, unchanged.
        """
        fn = getattr(self.module, 'traj_feature_dim', None)
        return int(obs_dim) if fn is None else int(fn(env_name))

    def action_dims(self, env):
        """Per-dimension choice count, for a MULTI-DISCRETE action space.

        None on every suite whose action is one categorical or one continuous
        vector; ``(3, 3, 3, 3, 2, 2)`` on kinetix, where PPO's head is built
        from it (`source/studies/generalists/actors.py:head_for`).
        """
        fn = getattr(self.module, 'action_dims', None)
        return None if fn is None else fn(env)

    @property
    def env_configs(self):
        return self.module.ENV_CONFIGS

    def __repr__(self):
        return f"Suite({self.name!r}, {self.module.__name__})"

    def __getattr__(self, item):
        # `build_policy`, `task_noise_vectors`, `make_scoring_fn`,
        # `make_descriptor_scoring_fn`, `make_trajectory_scoring_fn`,
        # `make_fixed_seed_scoring_fn`, `descriptor_dim`, `rl_env_fns`.
        # Reached only for names this object does not already have.
        try:
            return getattr(self.module, item)
        except AttributeError as exc:            # pragma: no cover - typo guard
            raise AttributeError(
                f"suite {self.name!r} ({self.module.__name__}) has no "
                f"{item!r}") from exc


def _gymnax_env(env_name, episode_length, task_options=None):
    # `tasks.build_env` returns gymnax's own EnvParams with no task options --
    # what every caller received before the options existed -- and a
    # `tasks.TaskSpec` wrapping them under `task_mod=physics`, where a sub-task
    # is a multiplier on a named physics parameter rather than an offset.
    from source.envs import gymnax_classic as tasks
    env, env_params = tasks.build_env(env_name, episode_length, task_options)
    base = getattr(env_params, 'base_params', env_params)
    return (env, env_params,
            int(env.observation_space(base).shape[0]),
            int(env.action_space(base).n))


def _mjx_env(env_name, episode_length, task_options=None):
    from jax import random

    from source.envs import mjx as tasks_mjx
    env, spec = tasks_mjx.build_env(env_name, episode_length, task_options)
    # A fixed key, not one from the run's stream: reading the observation width
    # must not move the search. See the module docstring.
    obs_dim, action_dim = tasks_mjx.env_dims(env, random.key(0))
    # The policy sees the sub-task vector too under --observe_task, so the
    # width it is built for has to include it.
    if getattr(spec, 'observe_task', False):
        obs_dim += int(spec.dim or 0)
    return env, spec, obs_dim, action_dim


def _minigrid_env(env_name, episode_length, task_options=None):
    from source.envs import minigrid as tasks_minigrid
    env, spec = tasks_minigrid.build_env(env_name, episode_length, task_options)
    obs_dim, action_dim = tasks_minigrid.env_dims(env)
    return env, spec, obs_dim, action_dim


def _kinetix_env(env_name, episode_length, task_options=None):
    from source.envs import kinetix as tasks_kinetix
    env, spec = tasks_kinetix.build_env(env_name, episode_length, task_options)
    obs_dim, action_dim = tasks_kinetix.env_dims(env)
    return env, spec, obs_dim, action_dim


def _suites():
    """Built on demand: importing a suite imports its simulator."""
    from source.envs import gymnax_classic as tasks
    return {
        'gymnax': Suite('gymnax', tasks, _gymnax_env),
    }


def get_suite(name):
    """The suite called ``name``. Everything but gymnax is imported lazily."""
    if name == 'mjx':
        from source.envs import mjx as tasks_mjx
        return Suite('mjx', tasks_mjx, _mjx_env)
    if name == 'minigrid':
        from source.envs import minigrid as tasks_minigrid
        return Suite('minigrid', tasks_minigrid, _minigrid_env)
    if name == 'kinetix':
        from source.envs import kinetix as tasks_kinetix
        return Suite('kinetix', tasks_kinetix, _kinetix_env)
    suites = _suites()
    if name not in suites:
        raise KeyError(f"unknown suite {name!r}; have "
                       f"{sorted(list(suites) + ['mjx', 'minigrid', 'kinetix'])}")
    return suites[name]


# Which suite owns which environment. Written out rather than discovered by
# importing both modules, because importing the mjx one pulls in MJX and the
# playground registry -- seconds, and a hard dependency for anyone who only
# wants the gymnax half. `train_all.py` reads this before it sets
# CUDA_VISIBLE_DEVICES, so it must stay import-free of jax as well.
ENV_SUITE = {
    'CartPole-v1': 'gymnax',
    'Acrobot-v1': 'gymnax',
    'MountainCar-v0': 'gymnax',
    'CheetahRun': 'mjx',
    'ant': 'mjx',
    # MiniGrid on xminigrid, since 2026-09-07; the pair of environments is a
    # task option (`envs=A,B`), see tasks_minigrid.ENV_CONFIGS.
    'MiniGrid': 'minigrid',
    'MiniGrid-L1024': 'minigrid',      # a 1024-step scan, for the 16x16 rooms
}

# Kinetix, since 2026-09-09: a sub-task is one of the twenty hand-designed
# MEDIUM levels. `Kinetix20` is the continual chain over all of them and
# `Kinetix-<level>` the stationary run on one -- twenty-one cells, listed from
# the suite's own table rather than retyped, because the level list is what
# `source/envs/kinetix.py` asserts against the level files on disk.
#
# The names come from `source/envs/kinetix_levels.py`, which imports NOTHING --
# the suite module itself pulls in jax, jax2d and the Kinetix renderer, and
# this table has to stay readable before the device mask is set.
from source.envs.kinetix_levels import CELLS as _KINETIX_CELLS   # noqa: E402

ENV_SUITE.update({name: 'kinetix' for name in _KINETIX_CELLS})
# DeepSea<N>-bsuite (gymnax_classic.DEEPSEA_ENV_NAMES); listed here by name so
# this table stays import-free, as the three classic entries are.
ENV_SUITE.update({f'DeepSea{n}-bsuite': 'gymnax' for n in (8, 10, 12, 14, 16, 20)})

ENV_NAMES = list(ENV_SUITE)


def make_env_for_run(config):
    """Rebuild a finished run's environment from its recorded config.

    ``(env, env_params, obs_dim, action_dim)``, exactly what ``make_env``
    returned to the trainer: the same environment, and the same meaning for a
    row of the run's ``noise_vectors``. The analysis scripts that score saved
    parameters -- the landscapes, the plasticity probes -- used to build a
    gymnax env themselves and treat every row as an observation offset, which
    is right for every run made before 2026-09-05 and silently wrong for a
    physics run, whose rows are multipliers: ``obs + [2.0]`` broadcasts and
    scores an offset nobody trained on. This reads ``config['task']`` -- the
    suite's ``describe()``, absent on an obs-noise gymnax run -- and passes
    its options back in.
    """
    task = config.get('task') or {}
    options = {}
    if task:
        options['task_mod'] = task['task_mod']
        for group, values in (task.get('options') or {}).items():
            # `describe()` nests the group's settings under the group's name
            # (`physics: {param, order, low, ...}` on gymnax, `friction: {...}`
            # on the ant); `build_env` reads them back as `<group>_<key>`.
            if isinstance(values, dict):
                for key, value in values.items():
                    options[f'{group}_{key}'] = value
            else:
                options[group] = values
    # A run that whitened its observations (`obs_norm`, the mjx NE arms) is
    # rebuilt whitened, so its saved weights read the input they were trained
    # on. Only passed when true: the other suites' `build_env` do not take it,
    # and a run on them that claimed it should fail loudly rather than be
    # rebuilt un-whitened.
    if config.get('obs_norm'):
        options['obs_norm'] = True
        # The statistics the run recorded, when it recorded them (NE runs since
        # 2026-09-13): rebuilding from them is exact, re-measuring them does
        # not reproduce (see `source/envs/mjx.py`, build_env).
        if config.get('obs_mean') is not None and config.get('obs_std') is not None:
            options['obs_mean'] = config['obs_mean']
            options['obs_std'] = config['obs_std']
    suite = get_suite(suite_for(config['env']))
    return suite.make_env(config['env'], int(config['episode_length']),
                          options or None)


def suite_for(env_name):
    """The suite that owns ``env_name``, as a name. Raises on an unknown env."""
    if env_name not in ENV_SUITE:
        raise KeyError(f"unknown environment {env_name!r}; have "
                       f"{sorted(ENV_SUITE)}")
    return ENV_SUITE[env_name]


def threshold_for(env_name, threshold_set='generalists'):
    """The solved threshold for ``env_name``, or None where there is not one.

    One place decides, so a table, a figure and a trainer cannot disagree about
    whether an environment has a threshold at all. gymnax has three named sets
    and this picks one of them; the mjx bodies have none --
    ``source/metrics/evaluation_metrics.py`` says outright that
    CheetahRun has no threshold to compare against, and the ant has none either
    -- so this returns None and everything downstream reports the generalist
    SCORE instead of a found/held count. Nothing invents a constant.
    """
    # The mjx bodies and MiniGrid have none -- see their modules.
    if suite_for(env_name) != 'gymnax':
        return None
    from source.envs.gymnax_classic import get_threshold
    return get_threshold(env_name, threshold_set)
