"""The brax body in this study: the reward, the offset, the friction, the build.

The body is brax's ``halfcheetah``. What is here is everything a sub-task
does to it: the speed-tracking reward, the observation offset, the
ground-friction sequences, the sub-task revisit rule, and the factory that
composes them.

## The reward

The cheetah runs ``TargetSpeedWrapper``: bounded per-step credit in [0, 1]
scaled by ``DEFAULT_SPEED_WEIGHT``, peaking at the sub-task's target speed,
rather than brax's stock ``halfcheetah`` reward, which is unbounded forward
velocity minus a control cost.

The substitution ``reward - x_velocity + speed_reward`` is exact, and that is a
fact about brax rather than an approximation: the cheetah's ``reward_run`` is
*exactly* ``x_velocity`` (``reward_run == x_velocity`` to float equality,
and ``reward == reward_run + reward_ctrl``), so subtracting
the velocity removes the whole forward term and leaves the control cost
untouched.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from brax import envs


# The physics backend passed to envs.create: 'mjx' rather than brax's
# 'generalized' default, so the friction sub-tasks measure the ground rather
# than the solver -- the generalized backend's coarse contact model returned
# NaN at x5.0 friction and let a policy slide at implausible speeds at x0.2.
DEFAULT_BACKEND = 'mjx'


# Friction multipliers, cycled default -> low -> high. Finite at every
# multiplier from x0.2 to x5.0 on mjx.
DEFAULT_FRICTION_MULT = 1.0

DEFAULT_LOW_MULT = 0.2      # slippery

DEFAULT_HIGH_MULT = 5.0     # sticky; NaNs on the generalized backend, not on mjx



class TargetSpeedWrapper:
    """Rewards running at `target_speed` rather than as fast as possible.

    brax halfcheetah's reward is `v_x - ctrl_cost`, i.e. unbounded in forward
    speed. This replaces the `v_x` term with a bounded credit peaking AT the
    target and falling off on both sides.

    Why a target speed
    ------------------
    Speed is a scalar the body cannot re-route around: a gait tuned for one
    speed does not run at another, where a change to the ground can be
    absorbed by a different gait that still moves forward.

    Precedent, in two places:
      * C-CHAIN (Tang et al., ICML 2025 -- inspiration/C-CHAIN) builds its
        Continual DMC benchmark by chaining tasks that differ only in target
        speed (walker stand/walk/run, quadruped walk/run/walk). That paper is
        about mitigating plasticity loss, so PPO demonstrably loses plasticity on
        a speed sequence -- and `cchain` is already a baseline in this repo.
      * MAML/PEARL's HalfCheetah-Vel uses exactly `-|v - v_target|`.

    Symmetric, deliberately. dm_control's walk/run reward is one-sided: full
    credit at or above the target, so a run-capable policy also satisfies walk
    and only walk->run is a real shift. `-|v_x - target|` is non-stationary in
    both directions -- a fast gait fails a slow target and a slow gait fails a
    fast one -- which is what makes a revisit to an earlier target a genuine
    re-adaptation rather than a freebie.

    Implemented by correcting the reward rather than by forking the env: brax
    puts `x_velocity` in `state.metrics` and the cheetah's `reward_run` term is
    exactly `x_velocity`, so subtracting one and adding the other is exact. That
    keeps the control cost untouched, so this composes with the friction
    scaling.
    """

    def __init__(self, env, target_speed, margin=None, weight=None):
        self._env = env
        self._target_speed = float(target_speed)
        # Gaussian width. Proportional to the target by default, so both
        # sub-tasks are equally forgiving: standing still earns 0.135 of the
        # credit whether the target is 0.5 or 2.0. A FIXED width cannot do that
        # -- at sigma 1.0 a standing body already earns 0.882 of the maximum for
        # target 0.5, leaving only 0.118/step to be gained by actually tracking.
        self._margin = float(margin) if margin is not None else (
            DEFAULT_SPEED_MARGIN_RATIO * abs(float(target_speed)))
        # How much the tracking term is worth relative to the rest of the
        # reward. See DEFAULT_SPEED_WEIGHT.
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
        # step. The cheetah names its forward term `reward_run`.
        for key in ('reward_run',):
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
        1.0 a standing body saw credit 0 AND gradient 0 for every velocity below
        2.0 -- an exploration dead zone, and PPO duly converged to standing
        still. A Gaussian is never
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



# Target speeds, cycled through by `speed_cycle`. Two targets alternating on one
# body, as C-CHAIN's Continual Quadruped alternates walk/run.
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

# Weight on the tracking term. At 5 the per-step gain from tracking perfectly
# rather than standing still is 4.32, so the objective is unambiguously the
# speed.
DEFAULT_SPEED_WEIGHT = 5.0



def speed_cycle(num_tasks, targets=DEFAULT_SPEED_TARGETS):
    """The target speed of each sub-task, cycling through `targets`."""
    targets = [float(t) for t in targets]
    return [targets[i % len(targets)] for i in range(int(num_tasks))]



def effective_task_idx(task_idx, task_period):
    """`task_idx` folded into the first `task_period` sub-tasks.

    The brax counterpart of `cycle_task_sequence` in
    source/utils/task_sequence.py, and the same design for the same reason:
    a sub-task the learner has already solved has to come back if forgetting is
    to be measured at all, and a flat modulo makes the gap between a sub-task
    and its revisit a constant of the design (`task_period`) rather than a
    per-task nuisance.

    Only the observation offset needs this. The friction and speed sequences
    are already cycles over a handful of values, so they revisit on their own;
    the offset is drawn fresh per (seed, task_idx) and would otherwise never
    repeat.

    A period of 0, None, or one at least as long as the run leaves the index
    untouched.
    """
    if not task_period or int(task_period) <= 0:
        return int(task_idx)
    return int(task_idx) % int(task_period)



class ObsOffsetWrapper:
    """Wrapper that adds a fixed per-sub-task offset vector to the observation.

    The gymnax continual protocol on brax: every sub-task after the first
    perturbs the observation with a FIXED vector drawn once per (trial, sub-task) --
    sensor miscalibration, not noise. The policy's inputs shift; the physics,
    the reward and the optimal behaviour do not. Nothing in the observation
    marks that a shift happened or what it is: a perturbation the policy can
    read off its inputs is a perturbation it can condition on.

    The offset is seeded off (seed, task_idx) alone -- the same guarantee the
    friction sequence gives: every method at a trial faces the same
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

def friction_cycle(num_tasks, default_mult=DEFAULT_FRICTION_MULT,
                   low_mult=DEFAULT_LOW_MULT, high_mult=DEFAULT_HIGH_MULT):
    """The friction multiplier of each sub-task, cycling default -> low -> high.

    Deterministic and identical for every method, trial and seed, so two
    methods at the same trial face the same ground at the same point in their
    budget. Sub-task 0 is the default multiplier, so the ground of the first
    sub-task is the noncontinual control's ground.
    """
    cycle = [float(default_mult), float(low_mult), float(high_mult)]
    return [cycle[i % len(cycle)] for i in range(int(num_tasks))]



def random_friction_sequence(rng_key, num_tasks, low_mult=DEFAULT_LOW_MULT,
                             high_mult=DEFAULT_HIGH_MULT,
                             default_mult=DEFAULT_FRICTION_MULT):
    """Log-uniformly sampled friction multiplier per sub-task.

    The Dohare et al. (Nature 2024) protocol: every sub-task's ground is a
    fresh sample rather than a revisit of three fixed values. With 3 recurring
    multipliers a revisit is a task the weights have already covered, and the
    learner can settle into one compromise policy for the whole cycle.

    Log-uniform rather than uniform because the multiplier acts as a scale:
    x0.2 and x5.0 are equally far from x1.0, while uniform sampling on
    [0.2, 5.0] would make four in five sub-tasks stickier than default.

    Sub-task 0 is pinned to `default_mult`, so the run starts on the
    unperturbed ground of the noncontinual control.
    """
    lo, hi = jnp.log(low_mult), jnp.log(high_mult)
    mults = jnp.exp(jax.random.uniform(
        rng_key, (int(num_tasks),), minval=lo, maxval=hi))
    sequence = [float(m) for m in mults]
    if sequence:
        sequence[0] = float(default_mult)
    return sequence



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

    Wrapper order matters: the speed correction reads the velocity of the state
    the physics actually produced, and the observation offset lands outermost,
    on the final observation.

    ``wrap=False`` for the RL trainer, which runs the env through brax's own
    ``training.wrap``. With ``wrap=True`` on both sides the stack carries two
    EpisodeWrappers and two AutoResetWrappers, and brax's Evaluator then
    averages over twice the true episode count -- which silently halved every
    reported RL return relative to NE.
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
