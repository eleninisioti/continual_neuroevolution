"""What every brax body in this study shares: the reward, the offset, the build.

Split out of ``source/envs/brax_ant.py`` on 2026-09-08, when the cheetah moved
from mujoco_playground's dm_control ``CheetahRun`` onto brax's ``halfcheetah``
and the two bodies stopped being one body plus a special case. What is here is
everything that mentions no limb: the speed-tracking reward, the observation
offset, the sub-task revisit rule, and the factory that composes them.
``brax_ant.py`` keeps the leg damage, the motor flip and the friction/leg
cycles, and imports the rest from here.

## One reward convention across bodies

Both bodies run ``TargetSpeedWrapper``, so an ant return and a cheetah return
mean the same thing: bounded per-step credit in [0, 1] scaled by
``DEFAULT_SPEED_WEIGHT``, peaking at the sub-task's target speed. That is what
makes a figure with both bodies on it readable, and it is why the cheetah did
not simply inherit brax's stock ``halfcheetah`` reward, which is unbounded
forward velocity minus a control cost and would put the two bodies on
incomparable scales.

The substitution ``reward - x_velocity + speed_reward`` is exact on both, and
that is a fact about brax rather than an approximation: the ant's
``reward_forward`` and the cheetah's ``reward_run`` are each *exactly*
``x_velocity`` (verified 2026-09-08 -- ``reward_run == x_velocity`` to float
equality, and ``reward == reward_run + reward_ctrl``), so subtracting the
velocity removes the whole forward term and leaves the survival bonus and the
control cost untouched.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from brax import envs


# The physics backend passed to envs.create. See the module docstring: 'mjx'
# rather than brax's 'generalized' default, so the ant and the cheetah share a
# contact model, and so the friction sub-tasks measure the ground rather than
# the solver. Every ant tree dated on or before 2026-07-30 predates this.
DEFAULT_BACKEND = 'mjx'



class TargetSpeedWrapper:
    """Rewards running at `target_speed` rather than as fast as possible.

    brax ant's reward is `v_x + healthy - ctrl_cost`, i.e. unbounded in forward
    speed. This replaces the `v_x` term with `-|v_x - target|`, so the objective
    peaks AT the target and falls off on both sides.

    Why a target speed and not more friction or more damage
    ------------------------------------------------------
    Friction and single-leg damage both turned out to be weak shifts for this
    robot, and the reason is the same for both: an ant has four legs and many
    gaits that walk forward, so it re-routes around a change to the ground or to
    one limb. In the 2026-07-30 12-sub-task run PPO's per-cycle retention was
    1.30 -- no degradation at all. Speed is a scalar the body cannot re-route
    around: four legs do not let a 1 m/s gait run at 5 m/s.

    Precedent, in two places:
      * C-CHAIN (Tang et al., ICML 2025 -- inspiration/C-CHAIN) builds its
        Continual DMC benchmark by chaining tasks that differ only in target
        speed (walker stand/walk/run, quadruped walk/run/walk). That paper is
        about mitigating plasticity loss, so PPO demonstrably loses plasticity on
        a speed sequence -- and `cchain` is already a baseline in this repo.
      * MAML/PEARL's HalfCheetah-Vel and Ant-Vel use exactly `-|v - v_target|`.

    Symmetric, deliberately. dm_control's walk/run reward is one-sided: full
    credit at or above the target, so a run-capable policy also satisfies walk
    and only walk->run is a real shift. `-|v_x - target|` is non-stationary in
    both directions -- a fast gait fails a slow target and a slow gait fails a
    fast one -- which is what makes a revisit to an earlier target a genuine
    re-adaptation rather than a freebie.

    Implemented by correcting the reward rather than by forking the env: brax
    ant puts `x_velocity` in `state.metrics` and its `reward_forward` term is
    exactly `x_velocity`, so subtracting one and adding the other is exact. That
    keeps the healthy bonus, the control cost and the termination rule untouched,
    so this composes with LegDamageWrapper and with the friction scaling.
    """

    def __init__(self, env, target_speed, margin=None, weight=None):
        self._env = env
        self._target_speed = float(target_speed)
        # Gaussian width. Proportional to the target by default, so both
        # sub-tasks are equally forgiving: standing still earns 0.135 of the
        # credit whether the target is 0.5 or 2.0. A FIXED width cannot do that
        # -- at sigma 1.0 a standing ant already earned 0.882 of the maximum for
        # target 0.5, leaving only 0.118/step to be gained by actually tracking,
        # against a survival bonus of 1.0/step. Standing still was close to
        # optimal and PPO found it.
        self._margin = float(margin) if margin is not None else (
            DEFAULT_SPEED_MARGIN_RATIO * abs(float(target_speed)))
        # How much the tracking term is worth relative to the survival bonus.
        # At weight 1 the whole speed objective is worth no more per step than
        # simply staying upright, so the reward barely distinguishes a policy
        # that tracks from one that survives. See DEFAULT_SPEED_WEIGHT.
        self._weight = float(weight) if weight is not None else DEFAULT_SPEED_WEIGHT

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _retarget(self, state):
        # metrics carry the same velocity the inner reward was built from, so
        # this is a substitution and not a second estimate of the speed.
        x_vel = state.metrics['x_velocity']
        speed_reward = self._speed_reward(x_vel)
        reward = state.reward - x_vel + speed_reward
        # The target is a constant of the sub-task and lives in the run
        # config; the speed error is x_velocity minus it, so nothing is lost
        # by not logging either here.
        metrics = dict(state.metrics)
        # Only keys the body already has. brax's EpisodeWrapper accumulates
        # `episode_metrics` by iterating state.metrics against a dict built at
        # reset, so introducing a key mid-episode raises KeyError on the first
        # step. The ant names its forward term `reward_forward` (and mirrors it
        # as `forward_reward`); the cheetah names it `reward_run`. Guarding on
        # presence is what lets one wrapper serve both without a body flag.
        for key in ('reward_forward', 'forward_reward', 'reward_run'):
            if key in metrics:
                metrics[key] = speed_reward
        return state.replace(reward=reward, metrics=metrics)

    def _speed_reward(self, x_vel):
        """Bounded, two-sided speed-tracking credit in [0, 1] per step.

        dm_control's shape, which is what C-CHAIN's Continual DMC benchmark
        actually uses -- `rewards.tolerance` returns a per-step value in [0, 1]
        with graded partial credit as the target is approached, so an episode
        caps at `episode_length` no matter which target is in force. Their own
        Continual Quadruped scores (234-315 against a 1000 ceiling) sit in that
        partial-credit band rather than at a floor.

        The first version of this wrapper used raw `-|v_x - target|`, which is
        unbounded below. That put an unreachable target at a dead floor: at
        target 3.0 both PPO and DNS scored -11 and -19 against a ~1000 ceiling,
        so a third of the sequence discriminated nothing. Bounded credit with a
        margin keeps an ambitious target informative -- an agent at 1.5 m/s
        against a 3.0 target still scores above one at 0.5.

        Two-sided, unlike dm_control. Theirs gives full credit at or ABOVE the
        target, so a run-capable policy also satisfies walk and only
        walk -> run is a real shift; that asymmetry is why their quadruped
        sequence is walk -> run -> walk. Peaking AT the target makes every
        revisit a genuine re-adaptation in both directions.

        Gaussian rather than the linear ramp this started with, and that choice
        is what makes the fast sub-task learnable at all. A linear kernel hits
        exactly zero at `margin` and stays there, so with target 3.0 and margin
        1.0 a standing ant saw credit 0 AND gradient 0 for every velocity below
        2.0 -- an exploration dead zone. PPO duly converged to standing still and
        collecting the survival bonus, scoring 983/983/983 across five revisits
        of the slow target and a flat 500 on the fast one. A Gaussian is never
        exactly zero, so there is always a gradient pointing at the target from
        anywhere, while the peak stays sharp enough to separate the two
        sub-tasks. It is also dm_control's own default sigmoid.
        """
        return self._weight * jnp.exp(-0.5 * jnp.square(
            (x_vel - self._target_speed) / self._margin))

    def step(self, state, action):
        return self._retarget(self._env.step(state, action))

    def reset(self, rng):
        # reset's reward is 0 and its metrics are zeroed, so there is nothing to
        # re-target; going through _retarget anyway would write a spurious
        # -|0 - target| into the first step's reward.
        return self._env.reset(rng)



# Target speeds, cycled alongside the leg cycle exactly as the friction
# multipliers are. Period 3 against the leg cycle's 4: coprime, so 12 sub-tasks
# visit each (leg, target) pair once and every figure and metric built for the
# friction sequence applies unchanged.
#
# Two targets, not three: with two the cycle has period 2, which shares a factor
# with the 4-leg cycle, so the speed sequence is run on the healthy ant
# (--leg_order none) to keep the two axes from locking in phase. That also
# matches C-CHAIN's Continual Quadruped, which alternates walk/run on one body.
#
# Both are reachable. A PPO policy from the friction sweep sustained a measured
# 7.99 m/s mean (median 8.16, max 10.9) over a full 1000-step episode, so 3.0 is
# well inside the ant's range -- the earlier belief that 3.0 was unreachable was
# wrong, and the -11 scores it produced came from the unbounded reward paying
# -2.0/step for standing still against a +1 healthy bonus, which made falling
# over immediately the optimal policy. The bounded form cannot do that.
#
# 0.5 is a walk; 2.0 is roughly what a brax ant reaches when it maximises speed
# under the default unbounded reward (~3000 return = ~1000 survival + ~2000
# forward over 1000 steps). Both are far inside the ~8 m/s a trained policy was
# measured to sustain, so neither is a ceiling.
#
# DEFAULT_SPEED_MARGIN is the Gaussian's sigma, fixed rather than proportional so
# the two sub-tasks are equally forgiving. At sigma 1.0 a policy sitting at one
# target scores ~0.14 on the other -- a 7x gap between the two optima, which is
# enough to make a revisit a real re-adaptation while leaving usable gradient
# everywhere.
DEFAULT_SPEED_TARGETS = (0.5, 2.0)

# Gaussian sigma as a FRACTION of the target, not an absolute width -- see
# TargetSpeedWrapper.__init__ for why a fixed width made standing still nearly
# optimal on the slow sub-task.
DEFAULT_SPEED_MARGIN_RATIO = 0.5

# Weight on the tracking term, against a survival bonus of 1.0/step. At weight 1
# the per-step gain from tracking perfectly rather than standing still was
# 0.86 against a 1.0 survival bonus, so the reward paid roughly as much for
# staying upright as for doing the task; at 5 it is 4.32, and the objective is
# unambiguously the speed. The episode ceiling becomes
# episode_length * (1 + weight) rather than 2 * episode_length.
DEFAULT_SPEED_WEIGHT = 5.0



def speed_cycle(num_tasks, targets=DEFAULT_SPEED_TARGETS):
    """The target speed of each sub-task, cycling through `targets`."""
    targets = [float(t) for t in targets]
    return [targets[i % len(targets)] for i in range(int(num_tasks))]



def effective_task_idx(task_idx, task_period):
    """`task_idx` folded into the first `task_period` sub-tasks.

    The ant counterpart of `cycle_task_sequence` in
    source/studies/gymnax/continual_common.py, and the same design for the same reason:
    a sub-task the learner has already solved has to come back if forgetting is
    to be measured at all, and a flat modulo makes the gap between a sub-task
    and its revisit a constant of the design (`task_period`) rather than a
    per-task nuisance.

    Only the observation offset needs this. The leg, friction, gravity and
    speed sequences are already cycles over a handful of values, so they
    revisit on their own; the offset is drawn fresh per (seed, task_idx) and
    would otherwise never repeat.

    A period of 0, None, or one at least as long as the run leaves the index
    untouched, so the default reproduces every existing ant tree.
    """
    if not task_period or int(task_period) <= 0:
        return int(task_idx)
    return int(task_idx) % int(task_period)



class ObsOffsetWrapper:
    """Wrapper that adds a fixed per-sub-task offset vector to the observation.

    The brax-ant port of the gymnax continual protocol (source/studies/gymnax/
    continual_common.py): every sub-task after the first perturbs the
    observation with a FIXED vector drawn once per (trial, sub-task) --
    sensor miscalibration, not noise. The policy's inputs shift; the physics,
    the reward and the optimal behaviour do not. Nothing in the observation
    marks that a shift happened or what it is, which is the property the
    FingerSpin probe showed matters: a perturbation the policy can read off
    its inputs is a perturbation it can condition on.

    The offset is seeded off (seed, task_idx) alone -- the same guarantee the
    friction and leg sequences give: every method at a trial faces the same
    offsets at the same point in its budget. Sub-task 0 is unperturbed, so it
    reproduces the noncontinual control.

    Applied OUTSIDE every other wrapper, so the offset lands on exactly the
    observation the policy would otherwise have seen.
    """

    # Fold-in stream tag, distinct from the friction stream's (1).
    _STREAM = 2

    def __init__(self, env, sigma, seed, task_idx):
        self._env = env
        sigma = float(sigma)
        if sigma > 0.0 and int(task_idx) > 0:
            key = jax.random.fold_in(
                jax.random.fold_in(jax.random.key(int(seed)), self._STREAM),
                int(task_idx))
            self._offset = sigma * jax.random.normal(
                key, (int(env.observation_size),))
        else:
            self._offset = jnp.zeros(int(env.observation_size))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, rng):
        state = self._env.reset(rng)
        return state.replace(obs=state.obs + self._offset)

    def step(self, state, action):
        state = self._env.step(state, action)
        return state.replace(obs=state.obs + self._offset)

def scale_friction(env, mult):
    """Rescale ground friction on the System, once, in place.

    Not idempotent -- rescaling an already-scaled model compounds -- so this is
    applied to a freshly built env and never twice. Writes
    ``sys.geom_friction``, the array both backends read, rather than
    ``sys.mj_model``, which would change only what is rendered.
    """
    if float(mult) == 1.0:
        return env
    sys = env.unwrapped.sys
    env.unwrapped.sys = sys.replace(geom_friction=sys.geom_friction * float(mult))
    return env


def create_env(env_name, episode_length, backend=DEFAULT_BACKEND,
               target_speed=None, speed_margin=None, speed_weight=None,
               friction_mult=1.0, obs_noise_sigma=0.0, obs_noise_seed=0,
               task_idx=0, obs_task_period=0, wrap=True):
    """A brax body with this study's reward and sub-task perturbations.

    The body-agnostic half of ``brax_ant.create_env_with_damaged_leg``, and the
    factory the cheetah uses outright. Wrapper order is that function's, for the
    same reasons: the speed correction reads the velocity of the state the
    physics actually produced, and the observation offset lands outermost, on
    the final observation.

    ``wrap=False`` for the RL trainer, which runs the env through brax's own
    ``training.wrap``. With ``wrap=True`` on both sides the stack carries two
    EpisodeWrappers and two AutoResetWrappers, and brax's Evaluator then
    averages over twice the true episode count -- which silently halved every
    reported RL return relative to NE until it was found on the ant.
    """
    if wrap:
        env = envs.create(env_name, episode_length=episode_length,
                          auto_reset=True, backend=backend)
    else:
        env = envs.create(env_name, episode_length=None, auto_reset=False,
                          backend=backend)
    env = scale_friction(env, friction_mult)
    if target_speed is not None:
        env = TargetSpeedWrapper(env, target_speed, speed_margin, speed_weight)
    if obs_noise_sigma and float(obs_noise_sigma) > 0.0:
        env = ObsOffsetWrapper(env, obs_noise_sigma, obs_noise_seed,
                               effective_task_idx(task_idx, obs_task_period))
    return env
