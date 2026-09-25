"""The MJX task family -- CheetahRun and the brax ant -- for this study.

The mujoco/brax counterpart of ``tasks.py``, and deliberately the same shape:
a *task* is one environment plus a fixed vector drawn from the trial index
alone, so every method at a given trial faces identical sub-tasks. What the
vector IS differs by body, and that is the one thing this module adds over the
gymnax side:

    obs_noise   CheetahRun. The vector is an offset on the observation and the
                policy sees ``obs + vector`` -- section C's construction and the
                paper's cheetah obs-noise block.
    friction    the ant. The vector is one number, a multiplier on the ground
                friction, and the PHYSICS changes while the observation does
                not -- the paper's own ant continual block
                (``source/envs/brax_ant.py``), whose sub-tasks are
                friction rescalings on a healthy ant at a target speed.

Everything env-facing is imported rather than re-implemented. The environment
comes from the paper's own factories (``source.envs.brax_common`` /
``source.envs.brax_ant``), the policy is
``source.algorithms.networks.ContinuousMLPPolicy`` -- the one the
paper's four NE trainers and its PPO share -- the offset draw is
``tasks.task_noise_vectors``, the same function the gymnax side calls, and the
friction values are the paper's ``friction_cycle`` / ``random_friction_sequence``.
What is new here is only the rollout wiring: a scoring function of the shape
this study's searchers want, ``score(genomes, key, task_vector) -> (pop,)``.

## Why the sub-task is a traced argument and not a rebuilt environment

The paper's continual runs perturb the observation with a wrapper and rescale
friction on a freshly built System, both once per sub-task. That is right for a
12-sub-task run that rebuilds its env at each boundary anyway. It is wrong
here: this study switches sub-task every ``task_interval`` generations inside
one jitted loop, and a rebuild would force a fresh MJX JIT at each of the
twenty boundaries.

The offset is therefore added where the policy reads the observation --
arithmetically the identical perturbation, since the offset enters neither the
dynamics nor the reward -- and the friction multiplier is applied to the
System INSIDE the traced rollout: ``env.unwrapped.sys`` is swapped for
``sys.replace(geom_friction=base * mult)`` for the duration of the trace and
restored after. That is brax's own idiom -- its
``DomainRandomizationVmapWrapper`` does exactly this to vmap a rollout over
batched Systems -- and it is what makes the multiplier a traced *argument*, so
the whole switching run compiles once and the joint schedule can ``vmap`` a
population's rollout over the two sub-tasks' frictions. ``check_friction`` in
``scripts/outdated/generalists/check_mjx_tasks.py`` asserts that a rollout under a
traced multiplier returns exactly what a statically rescaled env returns.

## The ant's reward is the paper's continual ant reward

``target_speed`` 2.0, the value ``continual_ant_friction_t24`` pins for every
sub-task. brax ant's stock reward is unbounded forward velocity, and the
paper's own measurement is that under it an ant re-routes around a friction
change -- four legs and many gaits walk forward -- so friction alone was not a
shift. Speed-tracking (``TargetSpeedWrapper``) is what made the paper's ant
friction block a benchmark, and the ant's sub-tasks here are read against
that block, so it carries the same reward. ``target_speed`` is a per-env
config field; None is brax's stock reward.

## No solved threshold

``evaluation_metrics.py`` says it outright: CheetahRun has no threshold to
compare against, and neither does the ant. So ``solved_threshold`` is None
here, and the analysis for this suite reports the generalist SCORE rather than
a found/held count -- see ``scripts/outdated/generalists/analysis/summarize_2task.py``
``schedule_stats_scores``. Nothing downstream invents a constant.

## Sigma does not transfer between bodies

``source/envs/brax_common.ObsOffsetWrapper`` measured it: the ant's 27
observation dims have a median per-dim std of 1.05 and dm_control's CheetahRun
had 0.36, so the ant block's sigma 2.0 was ~2x the ant's spread and ~5.5x that
cheetah's, and 0.7 was the cheetah value matching the ant's
perturbation-to-spread ratio.

THOSE TWO CHEETAH NUMBERS ARE STALE and describe a body this file no longer
builds. Re-measured on 2026-09-09 over 200 random-action steps, brax's
halfcheetah has a median per-dim std of 0.803 against the brax ant's 0.804 --
the two bodies now have essentially the SAME observation spread, so sigma 2.0
is ~2.5x it on both and needs no per-body correction. That is why
``CheetahRun`` keeps `noise_range` 2.0, which is also the value
``runs_repro2/mujoco/continual`` was made at.

``noise_range`` per environment records the value that environment's own runs
were made at, and is what ``--noise_range`` overrides. It only applies under
``task_mod: obs_noise``.
"""

from __future__ import annotations

import os

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from source.envs.gymnax_classic import task_noise_vectors  # noqa: F401 (re-exported)
from source.algorithms.networks import (
    create_continuous_policy_network,
    unflatten_params,
)
from source.metrics.aurora import episode_relative_indices

__all__ = [
    'ENV_CONFIGS', 'TaskSpec', 'build_env', 'build_policy', 'descriptor_dim',
    'make_scoring_fn', 'make_fixed_seed_scoring_fn',
    'make_descriptor_scoring_fn', 'make_trajectory_scoring_fn',
    'task_noise_vectors', 'task_vectors', 'obs_offset', 'rl_env_fns',
]


# Per-environment settings, pinned to what the paper's own runs used so the two
# sets of numbers stay comparable. `hidden_dims` is POLICY_ARCH's for the body;
# `noise_range` is the sigma that body's obs-noise runs were made at
# (`projects/neurips_2026_rebuttal/runs_repro2/{mujoco/continual,
# brax/continual_obsnoise}/*/config.json`).
#
# `task_mod` says what a sub-task vector means -- see the module docstring.
# `friction` is the paper's ant friction grid (`source/envs/brax_ant.py`):
# `mults` is its cycle default -> low -> high, so two sub-tasks are unperturbed
# ground and the slippery x0.2; `range` is the Slippery-Ant log-uniform draw its
# repro2 block used (`ANT_FRICTION_LOW_MULT=0.05` in
# `scripts/outdated/train/queue_repro2.sh`), reached with `--task_options
# friction_order=random`.
#
# `activation` is NOT a knob here: ContinuousMLPPolicy is tanh-hidden and it is
# what all four of the paper's NE trainers search on both bodies, so the study
# searches the same network they do. (POLICY_ARCH['brax'] records 'swish',
# which is the RL side's and what the dormancy criterion is chosen from -- the
# ant NE runs on disk record `activation: tanh` for exactly this reason.)
ENV_CONFIGS = {
    "CheetahRun": {
        # brax's `halfcheetah` since 2026-09-08, on the same MJX physics. It ran
        # on mujoco_playground's dm_control CheetahRun until then; moving it
        # deleted `source/studies/mujoco/` and `source/envs/mjx_cheetah.py` and
        # dropped the `mujoco_playground` dependency, which nothing else used.
        # Both bodies are 17-dim observation, 6 actuators, so the policy, the
        # searchers and the analysis are unchanged -- but the REWARD is not, so
        # no cheetah number from before that date is comparable and the block
        # is being re-run.
        "backend": "brax",
        "body": "halfcheetah",
        "hidden_dims": (128, 128),
        "episode_length": 1000,
        "task_mod": "obs_noise",
        "noise_range": 2.0,
        # dm_control's own `_RUN_SPEED` for CheetahRun, so the objective still
        # asks for the speed the replaced task asked for. Their credit is
        # ONE-sided (full marks at or above 10, linear ramp from 0); ours is
        # a two-sided Gaussian peaking AT the target, the ant's convention --
        # see `source/envs/brax_common.py` on why both bodies share it.
        #
        # PROVISIONAL: 10 is dm_control's number for dm_control's cheetah xml,
        # and brax's halfcheetah is a different model. It needs the same
        # treatment the ant's 2.0 got -- a specialist run confirming the target
        # is reachable and not trivially so -- before the block is read.
        "target_speed": 10.0,
        # The friction grid, so a cheetah sub-task can be the GROUND as well as
        # the observation. `scale_friction` is body-agnostic -- it rescales
        # `geom_friction` on whatever System it is handed -- so this is a
        # config entry and not a code path; without it `friction_multipliers`
        # raised KeyError on `f['low']` and the cheetah had one family only.
        #
        # The values are the ones the deleted `source/envs/mjx_cheetah.py`
        # cycled (x1.0 -> x0.2 -> x5.0, see block_mujoco_continual) and the
        # ant's `range` for the log-uniform draw. The cheetah is a planar body
        # with two feet and cannot re-route around a slippery ground the way
        # four legs can, so if anything this is a STRONGER shift here than on
        # the ant -- which is the open question, since the mujoco_playground
        # cheetah showed no RL degradation under it.
        "friction": {"order": "cycle", "default": 1.0, "low": 0.2, "high": 5.0,
                     "range": (0.05, 5.0)},
        "solved_threshold": None,
    },
    "ant": {
        "backend": "brax",
        "body": "ant",
        "hidden_dims": (128, 128),
        "episode_length": 1000,
        "task_mod": "friction",
        "friction": {"order": "cycle", "default": 1.0, "low": 0.2, "high": 5.0,
                     "range": (0.05, 5.0)},
        "noise_range": 2.0,
        "target_speed": 2.0,
        "solved_threshold": None,
    },
}

TASK_MODS = ('obs_noise', 'friction', 'action', 'speed')


class TaskSpec:
    """What a sub-task vector means for one built environment.

    This is the mjx suite's ``env_params``: ``make_env`` returns it, and every
    scoring function receives it, so the rollout can tell an observation offset
    from a friction multiplier without a flag threaded through the trainers.
    It is a Python object closed over by the jitted rollouts, never traced.

    ``base_sys`` is the System as built -- unperturbed ground -- and is what a
    friction multiplier is applied to, exactly once, at every trace. Applying
    it to whatever `sys` the env currently holds would compound, which is the
    hazard `source/envs/brax_ant.py` documents for the paper's static version.
    """

    def __init__(self, env_name, task_mod, options=None, base_sys=None,
                 obs_mean=None, obs_std=None, observe_task=False):
        if task_mod not in TASK_MODS:
            raise ValueError(f"task_mod must be one of {TASK_MODS}, got "
                             f"{task_mod!r}")
        self.env_name = env_name
        self.task_mod = task_mod
        # Privileged information, off unless --observe_task; see `augment`.
        self.observe_task = bool(observe_task)
        self.options = dict(options or {})
        self.base_sys = base_sys
        # OBSERVATION WHITENING, off unless build_env was asked for it.
        #
        # WHY IT IS HERE AT ALL. PPO on this body runs `normalize_obs: True`
        # (`_MJX_PPO`) and the NE path normalised nothing, so the two families
        # were reading differently-scaled inputs -- an asymmetry outside
        # everything `check()` polices, since the environment-step budgets are
        # identical. Measured on the ant, 600 random-action steps: per-dim std
        # runs 0.125 to 4.80, a 38x spread, with eight velocity dims at ~4-4.8
        # against thirteen at 0.13-0.40. Through a tanh first layer the large
        # dims dominate the pre-activations and the small ones do almost
        # nothing.
        #
        # STATIC, NOT RUNNING, AND THAT IS THE POINT. The standard OpenES
        # recipe keeps running statistics, but those make the objective
        # non-stationary -- the same genome scores differently at generation 10
        # and 300 -- which is a second change on top of the one being tested.
        # These are measured once, before the run, under a fixed seed, and
        # frozen. So this probe isolates INPUT SCALING and nothing else; if it
        # works, a running normaliser is the follow-up, not the conclusion.
        self.obs_mean = obs_mean
        self.obs_std = obs_std

    @property
    def dim(self):
        """Width of one sub-task vector: obs_dim for an offset, 1 otherwise.

        A friction multiplier and an action-reversal flag are both one number.
        """
        return 1 if self.task_mod in ('friction', 'action', 'speed') else None

    def augment(self, x, task):
        """The policy's input: the observation, plus the sub-task vector when
        this run was asked for it.

        A CONTROL, not an arm. Under `action` a memoryless policy cannot be a
        generalist over the two torque maps at all -- the same observation
        needs opposite torques -- so a run that fails WITHOUT this and succeeds
        WITH it has shown the obstacle to be observability rather than
        capacity, which is what says whether memory is worth building.

        Applied after whitening, so the whitening statistics keep the meaning
        they were measured with and the flag is not itself rescaled.
        """
        if not self.observe_task:
            return x
        t = jnp.reshape(jnp.asarray(task, dtype=x.dtype), (-1,))
        return jnp.concatenate([x, jnp.broadcast_to(t, x.shape[:-1] + t.shape)],
                               axis=-1)

    def act(self, action, task):
        """The action as the ENVIRONMENT receives it.

        The identity except under ``action``, where a sub-task REVERSES the
        action map. The gymnax family of the same name does `a -> n - 1 - a` on
        a discrete space (``FlipEnv`` in source/envs/gymnax_classic.py); the
        mirror of that on a symmetric torque box is `a -> -a`, so a policy that
        has learned a gait produces its exact opposite until it re-adapts. The
        observation and the physics are untouched, which is what makes this a
        different KIND of shift from the other two families: the body and the
        world are the same and only the meaning of the controls changed.

        ``task`` is a traced scalar, so a switching run still compiles once.
        """
        if self.task_mod != 'action':
            return action
        flip = jnp.reshape(task, (-1,))[0]
        return jnp.where(flip > 0.5, -action, action)

    def whiten(self, obs):
        """``(obs - mean) / std``, or ``obs`` untouched when off.

        Applied AFTER the sub-task's observation offset, so an `obs_noise`
        sub-task still shifts the raw observation and the shift is then scaled
        the same way the observation is -- otherwise the offset would mean
        something different under normalisation than without it, and the two
        arms of this probe would not be running the same experiment.
        """
        if self.obs_mean is None:
            return obs
        return (obs - self.obs_mean) / self.obs_std

    def obs_offset(self, task):
        """What is added to the observation before the policy reads it."""
        return task if self.task_mod == 'obs_noise' else 0.0

    @contextmanager
    def physics(self, env, task):
        """Run the enclosed trace under this sub-task's physics.

        A no-op for observation offsets. For friction, swaps the innermost
        env's System for one whose ``geom_friction`` is ``base * task[0]`` and
        restores the unperturbed System on exit -- including on an exception,
        so a failed trace cannot leave a tracer behind in a live object. The
        multiplier scales all three friction components of every geom, as the
        paper's static version does.
        """
        if self.task_mod == 'speed':
            # A sub-task is a TARGET SPEED. The reward wrapper bakes its target
            # at construction, so this swaps it for the duration of the trace
            # exactly as the friction branch swaps the System, and restores it
            # in `finally` so a failed trace cannot leave a tracer in a live
            # object. The margin is rescaled by the ratio the wrapper was BUILT
            # with, keeping both sub-tasks equally forgiving -- the property
            # TargetSpeedWrapper's proportional width exists to guarantee, and
            # without which standing still is near-optimal on the slow target.
            w = _find_speed_wrapper(env)
            if w is None:
                raise ValueError(
                    "task_mod 'speed' needs a TargetSpeedWrapper in the stack; "
                    'this cell was built with target_speed=None')
            old_t, old_m = w._target_speed, w._margin
            ratio = abs(old_m / old_t) if old_t else DEFAULT_SPEED_MARGIN_RATIO
            tgt = jnp.reshape(task, (-1,))[0]
            w._target_speed = tgt
            w._margin = ratio * jnp.abs(tgt)
            try:
                yield
            finally:
                w._target_speed, w._margin = old_t, old_m
            return
        if self.task_mod != 'friction':
            yield
            return
        inner = env.unwrapped
        base = self.base_sys
        mult = jnp.reshape(task, (-1,))[0]
        inner.sys = base.replace(geom_friction=base.geom_friction * mult)
        try:
            yield
        finally:
            inner.sys = base

    def describe(self):
        """For the run config: what the sub-task vectors were."""
        return {'task_mod': self.task_mod, 'options': dict(self.options),
                **({'observe_task': True} if self.observe_task else {})}


def build_env(env_name, episode_length, task_options=None):
    """The unperturbed environment plus its ``TaskSpec``, built once for the run.

    Once, not per sub-task: the sub-task is a traced argument to the rollout
    (see the module docstring), so the env is never rebuilt.

    ``task_options`` overrides the environment's task settings from the
    command line (``--task_options KEY=VALUE``): ``task_mod``, and for friction
    ``friction_order`` (``cycle`` / ``random``), ``friction_low``,
    ``friction_high``, ``friction_default``; for obs_noise nothing here --
    ``noise_range`` has its own flag. ``target_speed`` overrides the reward's
    target; ``none`` selects brax's stock reward.
    """
    cfg = ENV_CONFIGS[env_name]
    options = dict(task_options or {})
    task_mod = options.pop('task_mod', cfg['task_mod'])
    # Recorded by `describe()` for the log and superseded by low/high; a
    # finished run's task block hands it back through make_env_for_run.
    options.pop('friction_range', None)
    observe_task = bool(options.pop('observe_task', False))
    target_speed = options.pop('target_speed', cfg.get('target_speed'))
    if isinstance(target_speed, str) and target_speed.lower() == 'none':
        target_speed = None
    body = cfg["body"]
    if body == 'ant':
        # The ant, through the paper's own factory. With `target_speed` set this
        # is `source/envs/brax_ant.create_env_with_damaged_leg` on a healthy ant
        # at default friction and gravity -- the continual block's reward
        # wrapper, literally its code path -- and with None it is the stock brax
        # reward every noncontinual ant run used.
        from source.envs.brax_ant import create_env
        env = create_env(body, episode_length,
                         target_speed=(None if target_speed is None
                                       else float(target_speed)))
    else:
        # Every other brax body goes through the shared factory: no limb to
        # damage, so the ant's leg wrappers are not in the stack at all.
        from source.envs.brax_common import create_env
        env = create_env(body, episode_length,
                         target_speed=(None if target_speed is None
                                       else float(target_speed)))
    speed_targets = options.pop('speed_targets', cfg.get('speed_targets'))
    friction = dict(cfg.get('friction', {}))
    for key in ('order', 'low', 'high', 'default'):
        if f'friction_{key}' in options:
            friction[key] = options.pop(f'friction_{key}')
    # `range` is the body's stock draw window, recorded by `describe()` for
    # the log and superseded by `low`/`high`; a finished run's `task` block
    # hands it back through `registry.make_env_for_run`, where it is noise.
    options.pop('friction_range', None)
    # `obs_norm` arrives as an option from `registry.make_env_for_run` when a
    # FINISHED run recorded `obs_norm: true`, so the post-hoc passes rebuild
    # the same whitened input the NE arm was trained on. The training CLI
    # asks for it through NE_OBS_NORM instead (`source/studies/mjx/cli.py`),
    # because there one process may build the env for an RL arm that must
    # not whiten. Either route measures the same fixed-seed statistics.
    obs_norm = bool(options.pop('obs_norm', False))
    # The whitening statistics a finished run RECORDED (since 2026-09-13).
    # Rebuilding from them is exact; re-measuring them is not: the fixed-seed
    # random-policy rollout in `_measure_obs_stats` does not reproduce on MJX,
    # not even on one machine (std range 0.118-4.729 recorded vs 0.124-4.783
    # re-measured in the next process, home server, 2026-09-13). A whitened ant
    # NE agent rebuilt on re-measured statistics re-scored ~8% low.
    recorded_mean = options.pop('obs_mean', None)
    recorded_std = options.pop('obs_std', None)
    if options:
        raise ValueError(f'unknown task option(s) {sorted(options)}')
    spec = TaskSpec(env_name, task_mod,
                    dict(friction=friction, target_speed=target_speed,
                         **({'speed_targets': speed_targets}
                            if speed_targets is not None else {})),
                    base_sys=env.unwrapped.sys, observe_task=observe_task)
    if recorded_mean is not None and recorded_std is not None:
        spec.obs_mean = jnp.asarray(recorded_mean, dtype=jnp.float32)
        spec.obs_std = jnp.asarray(recorded_std, dtype=jnp.float32)
        print('  obs whitening  : ON  (statistics recorded by the run)', flush=True)
    elif obs_norm or os.environ.get('NE_OBS_NORM', '0') == '1':
        mean, std = _measure_obs_stats(env)
        # Host arrays: a device read of `mean` at finalise hung every long
        # NE run on the home server (futex_wait, 2026-09-18/19); a jit closes
        # over them as constants either way.
        spec.obs_mean, spec.obs_std = np.asarray(mean), np.asarray(std)
        print(f'  obs whitening  : ON  (per-dim std {float(std.min()):.3f}'
              f'-{float(std.max()):.3f} before, 1.0 after)', flush=True)
    return env, spec


def _measure_obs_stats(env, n_steps=600, seed=0, eps=1e-6):
    """Per-dim mean/std of the observation under a RANDOM policy, fixed seed.

    Random actions rather than the seed policy's: the statistics have to be
    independent of the search, or every arm would be whitened by a different
    transform and the arms would stop being comparable. Deterministic in
    `seed`, so two runs of the same config get the identical transform.

    `eps` floors the divisor: a dim that never moves would otherwise divide by
    zero and put inf into the first layer.
    """
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    key = jax.random.key(seed)
    state = reset(key)
    action_dim = env.action_size
    buf = []
    for _ in range(n_steps):
        key, sk = jax.random.split(key)
        buf.append(state.obs)
        state = step(state, jax.random.uniform(
            sk, (action_dim,), minval=-1.0, maxval=1.0))
    obs = jnp.stack(buf)
    return obs.mean(axis=0), jnp.maximum(obs.std(axis=0), eps)


def env_dims(env, key):
    """``(obs_dim, action_dim)`` for either backend, by resetting once.

    Jitted: an eager MJX reset is about a thousand separate op compiles and
    half a minute of startup, for a number that is read off a shape.
    """
    state = jax.jit(env.reset)(key)
    return int(state.obs.shape[-1]), int(env.action_size)


def friction_multipliers(spec, trial, num_tasks, first_task_clean=True):
    """The sub-task friction multipliers for a trial, as ``(num_tasks, 1)``.

    ``cycle`` is the paper's ``friction_cycle``: default, low, high, ... so the
    two-sub-task experiment is unperturbed ground against the slippery x0.2 and
    every trial faces the same pair -- a trial differs in its seed only, which
    is what makes eight trials eight seeds on ONE task pair rather than eight
    pairs of unknown separation. ``random`` is the paper's Slippery-Ant draw,
    ``random_friction_sequence`` on a key folded from the trial exactly as its
    trainers fold it from their seed, so a trial is its own multiplier and
    every method at that trial shares it.

    ``first_task_clean`` pins sub-task 0 to the default multiplier, which is
    what both paper functions already do; False lets the random draw perturb
    sub-task 0 as well.
    """
    from source.envs.brax_ant import friction_cycle, random_friction_sequence
    f = spec.options['friction']
    default = float(f.get('default', 1.0))
    if f.get('order', 'cycle') == 'cycle':
        seq = friction_cycle(num_tasks, default, float(f['low']),
                             float(f['high']))
    elif f['order'] == 'random':
        lo, hi = (float(x) for x in f.get('range', (0.05, 5.0)))
        key = random.fold_in(random.key(int(trial)), 1)
        if first_task_clean:
            seq = random_friction_sequence(key, num_tasks, lo, hi, default)
        else:
            seq = random_friction_sequence(key, num_tasks + 1, lo, hi,
                                           default)[1:]
    else:
        raise ValueError(f"friction order must be 'cycle' or 'random', got "
                         f"{f['order']!r}")
    return jnp.asarray(seq, dtype=jnp.float32)[:, None]


def _find_speed_wrapper(env):
    """The TargetSpeedWrapper in this stack, or None.

    Not ``env.unwrapped``: the speed wrapper sits mid-stack (above
    scale_friction, below the observation offset) and ``unwrapped`` returns the
    innermost brax env. Walks ``_env`` and tests ``vars`` rather than
    ``hasattr``, because the wrapper's ``__getattr__`` delegates to the env it
    wraps and would answer for attributes it does not own.
    """
    seen = 0
    while env is not None and seen < 16:
        if '_target_speed' in vars(env):
            return env
        env = getattr(env, '_env', None)
        seen += 1
    return None


def speed_sequence(spec, num_tasks):
    """The sub-task target speeds, as ``(num_tasks, 1)``.

    Cycles the cell's configured targets, so sub-task 0 is the FIRST target and
    the two-sub-task experiment is that pair -- deterministic, like the friction
    cycle and the action flip, so every trial faces the same targets and a trial
    differs in its seed alone. Not a random draw: with two states a draw would
    put consecutive sub-tasks in the same regime and give each trial a different
    number of real switches.

    The generalist here is genuine and graded, which is the point of this family:
    with targets 2.0 and 8.0 a policy running at either extreme is far outside
    the other's Gaussian window, while an intermediate gait scores moderately on
    both and so wins the worst case. Friction has no such trade-off -- four legs
    re-route around it -- which is why its retention is near zero.
    """
    raw = spec.options.get('speed_targets')
    if raw is None:
        from source.envs.brax_common import DEFAULT_SPEED_TARGETS
        targets = [float(t) for t in DEFAULT_SPEED_TARGETS]
    elif isinstance(raw, str):
        targets = [float(t) for t in raw.replace(',', ' ').split()]
    else:
        targets = [float(t) for t in raw]
    if not targets:
        raise ValueError('speed_targets is empty')
    seq = [targets[i % len(targets)] for i in range(int(num_tasks))]
    return jnp.asarray(seq, dtype=jnp.float32)[:, None]


def task_vectors(spec, trial, num_tasks, obs_dim, noise_range,
                 first_task_clean=True):
    """The sub-task sequence for a trial, as a ``(num_tasks, D)`` array.

    ``D`` is ``obs_dim`` for observation offsets -- the same draw as the gymnax
    side, ``tasks.task_noise_vectors`` -- and 1 for friction multipliers.
    ``suites.Suite.task_vectors`` is what calls this.
    """
    if spec.task_mod == 'friction':
        return friction_multipliers(spec, trial, num_tasks, first_task_clean)
    if spec.task_mod == 'action':
        return action_reversals(num_tasks)
    if spec.task_mod == 'speed':
        return speed_sequence(spec, num_tasks)
    return task_noise_vectors(trial, num_tasks, obs_dim, noise_range,
                              first_task_clean)


def action_reversals(num_tasks):
    """The sub-task action-reversal flags, as ``(num_tasks, 1)``.

    Sub-task 0 is the stock action map -- so it is the stationary experiment,
    as sub-task 0 is under the other two families -- and the flag then
    ALTERNATES: 0, 1, 0, 1, ... Not drawn from the trial seed, deliberately:
    a flag has two states, so a random draw would put consecutive sub-tasks in
    the same regime (boundaries at which nothing changes) and give each trial
    a different number of real switches. This is
    `source/utils/task_sequence.action_flip_sequence`, which the gymnax family
    uses for the same reason, on a continuous action space. A trial still
    differs in its training seed; it is the sub-task SEQUENCE that is shared.
    """
    return jnp.asarray([i % 2 for i in range(int(num_tasks))],
                       dtype=jnp.float32)[:, None]


def obs_offset(spec, task):
    """What the RL trainer adds to the observation for sub-task ``task``."""
    return spec.obs_offset(task)


def build_policy(key, obs_dim, action_dim, hidden_dims):
    """Return ``(policy, param_template, num_params)``.

    Same signature and same contract as ``continual_common.build_policy`` on
    the gymnax side, so ``train_nes`` calls one thing. The network is the
    shared ``ContinuousMLPPolicy``, i.e. the paper's cheetah/ant NE policy.
    """
    policy, param_template = create_continuous_policy_network(
        key, obs_dim, action_dim, tuple(hidden_dims))
    num_params = int(jax.flatten_util.ravel_pytree(param_template)[0].shape[0])
    return policy, param_template, num_params


def _rollout(env, spec, policy, param_template, episode_length, collect=False,
             whiten=False, trace=False):
    """``episode(flat_params, key, task) -> return``, or with the trace.

    One undiscounted episode return. Reward accrues only until the first
    termination and the scan always runs the full length, so the computation is
    a fixed shape and vmaps -- the same contract as the gymnax
    ``make_episode_fn``.

    ``task`` is one row of ``task_vectors``: an observation offset, added where
    the policy reads the observation, or a friction multiplier, applied to the
    physics for the duration of the trace (``TaskSpec.physics``).

    With ``collect`` it additionally returns the per-step observations and a
    validity mask, which is what the two descriptor paths reduce. That is a
    flag rather than something always returned and discarded: a 1000-step
    episode of a 17-dim observation vmapped over 512 x 3 rollouts is a 104 MB
    array, and only DNS ever reads one. The plain scoring path emits nothing
    from the scan at all.

    ``nan_to_num`` on the reward is the mjx guard the paper's trainers carry: a
    diverging sim returns NaN on the terminating step, and one NaN in a
    population poisons every reduction over it.

    ``trace`` is the post-hoc passes' variant of ``collect``: it emits the
    POLICY INPUT (offset and, when the spec carries whitening statistics,
    whitened) rather than the raw observation, plus the torso's xy after each
    step as an occupancy feature. See ``make_trace_fn``.
    """

    def episode(flat_params, key, task):
        params = unflatten_params(flat_params, param_template)
        offset = spec.obs_offset(task)
        with spec.physics(env, task):
            state = env.reset(key)

            def step_fn(carry, _):
                state, total, done_flag = carry
                obs = state.obs
                # `whiten` is a Python bool closed over at trace time, so the
                # off branch compiles to exactly the pre-whitening rollout.
                #
                # A PARAMETER, NOT A PROPERTY OF `spec`. It used to be the
                # latter, and that broke every RL row of runs_ant_v2 (ppo 4919
                # -> 220): run_ppo scores its policy through this same
                # function, on purpose, so both families share one ruler --
                # and its normaliser is already FOLDED into Dense_0 by
                # actors.fold_normalizer, so the folded network expects RAW
                # observations. Whitening on the shared spec normalised them a
                # second time. Only run_nes asks for it now.
                x = obs + offset
                x = spec.whiten(x) if whiten else x
                action = policy.apply(params, spec.augment(x, task))
                next_state = env.step(state, spec.act(action, task))
                # A diverging simulation ends the episode here and pays
                # nothing for the step that diverged -- the scoring-path twin
                # of `_reset_blown_up`. Without it an unbounded reward
                # (`target_speed=None`) makes breaking the solver the highest
                # scoring behaviour there is, and the search finds it.
                blown = _is_blown_up(next_state.obs)
                reward = jnp.nan_to_num(next_state.reward) * (1.0 - blown)
                total = total + reward * (1.0 - done_flag)
                valid = 1.0 - done_flag
                done_flag = jnp.maximum(done_flag,
                                        jnp.maximum(next_state.done, blown))
                if trace:
                    # Body 0 is the torso on every brax body; its xy is where
                    # the agent went.
                    emitted = (x, next_state.pipeline_state.x.pos[0, :2], valid)
                elif collect:
                    emitted = (obs, valid)
                else:
                    emitted = None
                return (next_state, total, done_flag), emitted

            (_, total, _), out = jax.lax.scan(
                step_fn, (state, 0.0, 0.0), None, length=episode_length)
        if trace:
            inputs, torso_xy, valid = out
            return total, inputs, torso_xy, valid
        if not collect:
            return total
        all_obs, valid = out
        return total, all_obs, valid

    return episode


def make_scoring_fn(env, env_params, policy, param_template, episode_length,
                    num_evals, whiten=False):
    """``score(genomes, key, task) -> (pop,)`` mean return.

    ``env_params`` is this suite's ``TaskSpec``; the gymnax function of the
    same name takes gymnax's params in the slot, so ``train_nes`` needs no
    branch.
    """
    episode = _rollout(env, env_params, policy, param_template, episode_length, whiten=whiten)

    def score(genomes, key, task):
        pop = genomes.shape[0]
        keys = random.split(key, pop * num_evals)
        repeated = jnp.repeat(genomes, num_evals, axis=0)
        returns = jax.vmap(episode, in_axes=(0, 0, None))(
            repeated, keys, task)
        return returns.reshape(pop, num_evals).mean(axis=1)

    return score


def make_fixed_seed_scoring_fn(env, env_params, policy, param_template,
                               episode_length, num_evals, eval_seed=0, whiten=False):
    """Like ``make_scoring_fn`` but with the *same* reset keys every call.

    For the landscape, and for the same reason as on gymnax: two nearby points
    must differ because the genomes differ, not because they drew different
    resets. Never use it for training.
    """
    episode = _rollout(env, env_params, policy, param_template, episode_length, whiten=whiten)
    eval_keys = random.split(random.key(eval_seed), num_evals)

    def score(genomes, task):
        per_eval = jax.vmap(
            lambda k: jax.vmap(episode, in_axes=(0, None, None))(
                genomes, k, task))(eval_keys)
        return per_eval.mean(axis=0)

    return score


def rl_env_fns(env, env_params, num_envs):
    """``(reset(key, task), step(key, state, action, task))`` for PPO.

    The vectorised environment interface ``train_ppo`` drives, in the shape
    the gymnax module's function of the same name has, so the trainer holds
    no branch on the backend. ``reset`` returns ``(obs, state)``; ``step``
    returns ``(obs, state, reward, done)``. Both run under the sub-task's
    physics. brax's ``AutoResetWrapper`` hands back the reset observation on
    the step that terminates, as gymnax does, so the two backends' transitions
    mean the same thing to GAE.

    The step's key is accepted for the shared signature and unused: the env
    is deterministic given the state, and brax's auto-reset resets to the
    state of the first reset rather than to a fresh draw. That is brax's
    convention and the paper's PPO ran under it.
    """
    spec = env_params

    def reset(key, task):
        with spec.physics(env, task):
            state = jax.vmap(env.reset)(random.split(key, num_envs))
        return spec.augment(state.obs, task), state

    def step(key, state, action, task):
        del key
        with spec.physics(env, task):
            state = jax.vmap(env.step)(state, spec.act(action, task))
        state = _reset_blown_up(state)
        return (spec.augment(state.obs, task), state,
                jnp.nan_to_num(state.reward), state.done)

    return reset, step


OBS_BLOWUP = 1e3        # |obs| beyond this is a diverging simulation, not a state


def _is_blown_up(obs):
    """1.0 where an observation says the simulation has diverged, else 0.0."""
    ok = jnp.all(jnp.isfinite(obs) & (jnp.abs(obs) < OBS_BLOWUP))
    return 1.0 - ok.astype(jnp.float32)


def _reset_blown_up(state):
    """Reset every env whose physics just produced a non-finite observation.

    MJX has no NaN protection. Once the contact solver diverges -- seen on the
    cheetah under the PPO policy after a ground-friction change, in 40 of 40
    trials, and never under the NE rollouts, which start every episode from
    reset -- that env's state is NaN for good: brax's auto-reset never fires,
    because ``done`` comes from the body's health and not from finiteness, and
    one NaN observation in a 4096-env window turns the PPO update, and with it
    every parameter, NaN. The blow-up is treated as a termination instead: the
    env goes back to the auto-reset wrapper's first-reset state (exactly what
    a ``done`` step does), its ``done`` is set so GAE closes the episode there,
    and that step's reward is zeroed. Every other env is bit-unchanged.

    The test is on MAGNITUDE, not only finiteness: a blow-up takes a few steps
    to reach inf, and the finite-but-astronomical observations on the way
    there are what poisoned the running observation normaliser (measured: one
    friction switch, one update, and the policy scored the untrained floor on
    every sub-task, all inputs squashed to ~0 by a variance in the 1e12s).
    Nothing a healthy body produces is anywhere near ``OBS_BLOWUP``: the
    observation carries no x position, and joint velocities are O(10).
    """
    if 'first_pipeline_state' not in state.info:
        raise ValueError("rl_env_fns needs brax's AutoResetWrapper in the "
                         "env stack (no first_pipeline_state in state.info)")
    bad = ~jnp.all(jnp.isfinite(state.obs)
                   & (jnp.abs(state.obs) < OBS_BLOWUP), axis=-1)

    def where_bad(x_reset, x):
        if bad.shape and bad.shape[0] != x.shape[0]:
            return x
        b = jnp.reshape(bad, [x.shape[0]] + [1] * (x.ndim - 1))
        return jnp.where(b, x_reset, x)

    pipeline_state = jax.tree_util.tree_map(
        where_bad, state.info['first_pipeline_state'], state.pipeline_state)
    obs = where_bad(state.info['first_obs'], state.obs)
    done = jnp.where(bad, jnp.ones_like(state.done), state.done)
    reward = jnp.where(bad, jnp.zeros_like(state.reward), state.reward)
    return state.replace(pipeline_state=pipeline_state, obs=obs, done=done,
                         reward=reward)


# ============================================================================
# The post-hoc passes (`source/envs/run_context.RunContext`)
# ============================================================================
#
# What `evaluate_continual`, `behavioural_divergence` and
# `plasticity_checkpoints` need beyond the training interface, as on MiniGrid
# and Kinetix. Two things are particular to this suite:
#
# WHITENING. An NE run trained with `--obs_norm` saved RAW policy weights that
# expect the whitened observation, while an RL run's `final` checkpoint has
# its normaliser folded into `Dense_0` and expects the raw one. The spec
# decides: `registry.make_env_for_run` builds the env with `obs_norm` when the
# run recorded it, so `spec.obs_mean` is set exactly for the runs that
# whitened, and every rollout below whitens iff it is set. That is also why
# `RunContext.cache_key` carries `obs_norm`: the two kinds of run must not
# share a jitted trace.
#
# STATES ARE POLICY INPUTS. The trace keeps what the policy consumed (offset
# added, whitening applied) rather than the raw observation, and `policy_input`
# is the identity. `behavioural_divergence` measures row i of its matrix on
# the states agent i visited on sub-task i, under sub-task i's shift -- which
# is exactly this input -- and `SuiteContext.logits_of` takes no task.

def make_trace_fn(env, env_params, policy, param_template, episode_length):
    """``trace(flat_params, key, task) -> (states, occupancy, alive, ret)``.

    ``states`` are the ``(T, obs_dim)`` policy inputs (see above),
    ``occupancy`` the torso's xy after each step, ``alive`` the mask of steps
    before the first termination, ``ret`` the undiscounted return -- the same
    return the training rollout scores.
    """
    spec = env_params
    episode = _rollout(env, spec, policy, param_template, episode_length,
                       whiten=spec.obs_mean is not None, trace=True)

    def trace(flat_params, key, task):
        total, inputs, torso_xy, valid = episode(flat_params, key, task)
        return inputs, torso_xy, valid, total

    return trace


def policy_input(states):
    """The trace already keeps the policy's input; nothing to encode."""
    return jnp.asarray(states)


_PROBE_FNS = {}


def _random_walk_fn(env, spec, steps_per_episode):
    """``walk(keys, task) -> (inputs, alive)``, jitted ONCE per built env.

    The sub-task is a traced argument, as in the training rollouts, so the
    hundred distinct sub-task rows of a ten-trial cell share one compile
    rather than paying an MJX trace each.
    """
    cache_key = (id(env), id(spec), int(steps_per_episode))
    if cache_key in _PROBE_FNS:
        return _PROBE_FNS[cache_key]
    whiten = spec.obs_mean is not None
    action_dim = int(env.action_size)

    def one(key, task):
        reset_key, act_key = random.split(key)
        offset = spec.obs_offset(task)
        actions = random.uniform(act_key, (steps_per_episode, action_dim),
                                 minval=-1.0, maxval=1.0)
        with spec.physics(env, task):
            state = env.reset(reset_key)

            def step_fn(carry, action):
                state, done_flag = carry
                x = state.obs + offset
                x = spec.whiten(x) if whiten else x
                emitted = (x, 1.0 - done_flag)
                next_state = env.step(state, action)
                done_flag = jnp.maximum(done_flag, next_state.done)
                return (next_state, done_flag), emitted

            _, (obs_seq, alive) = jax.lax.scan(step_fn, (state, 0.0), actions)
        return obs_seq, alive

    fn = jax.jit(jax.vmap(one, in_axes=(0, None)))
    _PROBE_FNS[cache_key] = fn
    return fn


def random_policy_observations(env, env_params, task, num_obs, seed=0,
                               steps_per_episode=32):
    """``(num_obs, obs_dim)`` policy inputs under UNIFORM RANDOM actions in
    [-1, 1] on sub-task ``task``, the dormancy probe batch. Many short episode
    prefixes rather than one long walk, as on the other suites: a random
    walk's states are autocorrelated and a probe wants coverage. Under the
    sub-task's physics and shift, so a friction sub-task's probe is drawn on
    that friction and an offset sub-task's probe carries that offset.
    """
    episodes = -(-num_obs // steps_per_episode)
    walk = _random_walk_fn(env, env_params, steps_per_episode)
    obs_seq, alive = walk(random.split(random.key(seed), episodes),
                          jnp.asarray(task))
    obs_seq = np.asarray(obs_seq).reshape(-1, obs_seq.shape[-1])
    alive = np.asarray(alive).reshape(-1) > 0
    live = obs_seq[alive]
    if len(live) < num_obs:
        live = np.concatenate([live, obs_seq[~alive]])[:num_obs]
    return jnp.asarray(live[:num_obs])


def descriptor_dim(env_name):
    """Width of the hand-crafted descriptor for an environment."""
    return 2


def handcrafted_descriptor(last_obs, env_name):
    """Where the episode ended, as two numbers.

    The hand-crafted alternative to AURORA, kept for the same reason the gymnax
    module keeps one: two interpretable numbers and a useful ablation. AURORA
    is the default and is what the paper's own cheetah DNS runs used
    (`descriptor: aurora` in their configs).

    The proper selection descriptor for the ant is the torso's final x/y, which
    the paper's DNS reads off the pipeline state rather than the observation;
    that is not reachable from ``obs`` alone, so ``--descriptor handcrafted``
    on the ant falls back to the first two observation coordinates and is an
    ablation rather than a reproduction. On the cheetah the paper's DNS has no
    hand-crafted descriptor at all.
    """
    return last_obs[:2]


def _last_valid(all_obs, valid):
    """The observation at the last step that was still part of the episode."""
    idx = jnp.maximum(jnp.sum(valid).astype(jnp.int32) - 1, 0)
    return all_obs[idx]


def make_descriptor_scoring_fn(env, env_params, policy, param_template,
                               episode_length, num_evals, env_name, whiten=False):
    """``score(genomes, key, task) -> (fitness, descriptors)``."""
    episode = _rollout(env, env_params, policy, param_template, episode_length,
                       collect=True, whiten=whiten)

    def one(flat_params, key, task):
        total, all_obs, valid = episode(flat_params, key, task)
        return total, handcrafted_descriptor(_last_valid(all_obs, valid),
                                             env_name)

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
                               episode_length, num_evals, traj_steps=50, whiten=False):
    """``score(genomes, key, task) -> (fitness, observations)``.

    The AURORA path, sub-sampled with ``episode_relative_indices`` -- the same
    reference function the gymnax side and the paper's cheetah DNS both use, so
    samples land inside the episode that happened rather than being spread over
    the episode CAP.

    Only the first evaluation's trajectory is kept when ``num_evals > 1``,
    matching the reference: averaging trajectories from different resets would
    describe no episode that happened.
    """
    num_traj_steps = min(traj_steps, episode_length)
    episode = _rollout(env, env_params, policy, param_template, episode_length,
                       collect=True, whiten=whiten)

    def one(flat_params, key, task):
        total, all_obs, valid = episode(flat_params, key, task)
        return total, all_obs[episode_relative_indices(valid, num_traj_steps)]

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
