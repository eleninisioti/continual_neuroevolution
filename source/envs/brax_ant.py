"""The brax ant: how one is built, and what a sub-task does to it.

Was ``source/studies/brax/continual_legs_common.py``, split in two so that an
environment is not something a study reaches into a trainer directory to get:
the wrappers, the factories and the sub-task cycles are here, while the flags
that select them, the task labels and the GIF rendering moved to
``source/studies/brax/cli.py``. ``create_env`` -- the healthy ant the noncontinual block
scores, which delegates here whenever a target speed is set -- came from
``source/studies/brax/train_GA_ant.py``, and it is the reason for the split: importing
that trainer to build an ant also ran its module-level ``CUDA_VISIBLE_DEVICES``
assignment. Nothing in this module imports a trainer or a parser.

A continual run is a
sequence of sub-tasks seen one after the other by a single learner that is never
told a switch happened. A sub-task here is *two* simultaneous perturbations: one
leg is disabled -- its actuators zeroed and its two joints pinned at a fixed
angle -- and the ground friction is rescaled by a global multiplier. Both change
at every boundary. The observation and action dimensions are unchanged (the dead
actuators still occupy their slots and the frozen joints still report their
angles), so one policy spans the whole sequence.

Why friction as well as damage
------------------------------
Leg damage alone was not a meaningful distribution shift for this robot. In the
2026-07-29 trial-1 sweep, PPO on the damaged continual sequence reached 4045 by
sub-task 4 against 4437 for PPO on the *healthy* noncontinual ant -- a 9% cost
for losing a limb -- and recovered from each switch inside ~13% of the sub-task's
step budget. Ant is kinematically redundant enough to walk on three legs, and
the joints are pinned at the midpoint of their range (see `_leg_indices`), which
is the mildest available lock. The sequence was therefore asking every method to
re-solve a near-identical problem twelve times, which is not a continual-learning
benchmark. Rescaling the ground underneath the damaged robot changes the contact
dynamics the surviving legs depend on, so a gait tuned for one sub-task does not
transfer to the next.

This also makes the ant block the direct counterpart of the cheetah block, whose
sub-tasks are pure friction rescalings, rather than a differently-shaped
experiment that happens to live in the same paper. Since 2026-07-30 the two run
on the same simulator as well (see "Which backend" below), so the ant's friction
grid is the cheetah's rather than an analogue of it.

Every number quoted above was measured on the `generalized` backend, and the
weakness of friction as a shift was partly a property of that solver -- it let
the ant slide at x0.2 rather than lose traction. Whether friction is a strong
shift on mjx is what the block under brax/continual_friction_mjx/ is measuring;
until it reports, treat the "friction is weak" finding as generalized-only.

Three properties matter for the comparison and are why this lives in one module
rather than in each trainer:

  * The sequence is a deterministic cycle, not a sample. Every method, at every
    trial, sees `leg 1 -> leg 2 -> leg 3 -> leg 4 -> leg 1 -> ...` crossed with
    `default -> low -> high -> default -> ...`. Nothing about it is drawn from
    the training RNG, so a GA run and a PPO run at the same trial face the same
    damage on the same ground at the same point in their budget and can be
    compared one for one.

    The two cycles have coprime periods (4 legs, 3 multipliers), so the combined
    period is 12 and the default `--num_tasks 12` visits each of the twelve
    (leg, friction) pairs exactly once. Each leg is still damaged three times,
    but on different ground each time, so no pair is over-represented and no
    sub-task is a repeat of an earlier one.

    `--leg_order random` restores the older sampled leg sequence, which is what
    train_RL_ant_continual_legs.py did before this module existed. It is kept
    for reproducing those runs and is not what the sweep uses.
    `--friction_order none` pins every multiplier to the default, which
    reproduces the damage-only sequence described above.
    `--friction_order random` replaces the cycle with a fresh log-uniform
    sample per sub-task (the Slippery-Ant protocol; see
    random_friction_sequence). Still shared across methods at a trial, because
    it is seeded off --seed rather than the training RNG.

  * Both perturbations are applied to a freshly created environment. Scaling
    `geom_friction` is not idempotent, so re-scaling an already-scaled model
    would compound; every switch rebuilds the env from scratch and applies the
    multiplier once. The env has to be rebuilt at each switch anyway because the
    scoring function is JIT-compiled against it, so this costs nothing extra.

  * The multiplier is applied to `sys.geom_friction`, not to `sys.mj_model`.
    Both backends read the System's own array -- brax's `System` subclasses
    `mjx.Model`, and the mjx pipeline hands `sys` straight to `mjx.step` --
    while mutating mj_model alone would change the rendered model and leave the
    physics untouched.

Which backend
-------------
`DEFAULT_BACKEND` is 'mjx', overridable per run with `--backend`. This is a
change of simulator, not a tuning knob: everything in projects/ under
brax/continual_*, brax/noncontinual and tuning/brax_ant* dated on or before
2026-07-30 was produced on brax's `generalized` backend, which is what
`envs.create('ant')` defaults to, and is not comparable step-for-step with an
mjx run.

The reason for the move is that the friction sequence was measuring the solver.
brax's generalized backend has a coarse contact model, and both of the
workarounds this module carried were artifacts of it: x5.0 friction returned NaN
(hence a 3.0 "sticky" value against the cheetah block's 5.0), and at x0.2 a PPO
policy scored HIGHER than on normal ground by sliding at implausible speeds,
which is what set the 2026-07-30 friction block aside. mjx is the same simulator
the cheetah block runs on, so the two bodies share a contact model and the
ant's friction sub-tasks are comparable to the cheetah's rather than merely
analogous. Since 2026-09-08 the cheetah is brax's `halfcheetah` on the same
MJX physics, so they are now literally the same stack.

Sub-task 0 is already damaged. Unlike the cheetah sequence, whose sub-task 0 is
the unperturbed environment and therefore reproduces the noncontinual control,
there is no healthy sub-task here: 12 sub-tasks over 4 legs divides evenly only
if every one of them damages a leg. Sub-task 0 does run at friction x1.0, so it
is the unperturbed *ground*. The healthy ant is the brax_noncontinual block,
which is the control this sequence is read against.
"""


from __future__ import annotations


import jax

import jax.numpy as jnp


from brax import envs

from source.envs.brax_common import (
    DEFAULT_BACKEND,
    ObsOffsetWrapper,
    TargetSpeedWrapper,
    effective_task_idx,
)


NUM_LEGS = 4


# Ant leg naming: leg L owns joints "hip_L" and "ankle_L" (L = 1..4).
LEG_NAMES = {None: 'HEALTHY', 0: 'LEG1', 1: 'LEG2', 2: 'LEG3', 3: 'LEG4'}


# Friction multipliers, cycled default -> low -> high alongside the leg cycle.
#
# The same 1.0 / 0.2 / 5.0 the cheetah block runs, which is possible only
# because the ant is now on mjx too. On brax's generalized backend the high
# value had to be 3.0: sweeping the multiplier under a fixed action sequence (8
# seeds, 300 steps) returned finite rewards from x0.1 to x4.0 and NaN at x5.0.
# The same sweep on mjx is finite at every multiplier from x0.2 to x5.0, so the
# cap was the solver rather than the robot and the two suites can share a
# friction grid. Running --backend generalized needs --friction_high_mult 3.0
# passed explicitly; nothing checks for it, and it will NaN if forgotten.
DEFAULT_FRICTION_MULT = 1.0

DEFAULT_LOW_MULT = 0.2      # slippery

DEFAULT_HIGH_MULT = 5.0     # sticky; NaNs on the generalized backend, not on mjx


FRICTION_CYCLE = ('default', 'low', 'high')


# Gravity multipliers, an axis independent of friction and cycled the same way.
#
# Why gravity at all: friction only changes the tangential limit at the contact,
# and four legs re-route around it -- the friction sequence has PPO improving
# across its first cycles rather than degrading. Gravity rescales every contact
# force and the body's own weight at once, which no gait can compensate away.
#
# Why the HEAVY side only (1.0 / 2.0 / 4.0 rather than something under 1.0):
# ant terminates outside `healthy_z_range` = (0.2, 1.0), and reducing gravity
# floats the torso up into that bound. Measured over 120 zero-action steps on
# mjx, the torso ends at z 0.70 at x0.5 and 0.64 at x0.25 against 0.55 unmodified,
# and the return collapses to 97 and 80 against 118 -- those are episodes ending
# early for being too HIGH, so a low-gravity sub-task would mostly measure the
# termination check rather than locomotion. The heavy side has no such artifact:
# the torso is progressively squashed (0.55 -> 0.51 -> 0.44 -> 0.36 at x1, x2,
# x4, x8) with the return flat at ~119, and every multiplier from 0.1 to 8.0 is
# finite.
DEFAULT_GRAVITY_MULT = 1.0

DEFAULT_GRAVITY_MID = 2.0    # heavy

DEFAULT_GRAVITY_HIGH = 4.0   # very heavy; x8 is finite too but squashes the ant


GRAVITY_CYCLE = ('default', 'mid', 'high')



def _leg_indices(mj_model, leg_idx):
    """Return (action_indices, qpos_indices, qvel_indices, locked_angles) for a leg.

    leg_idx is 0-based (0 -> joints hip_1/ankle_1).

    The action -> joint mapping is NOT leg-sequential (the XML declares the
    actuators as hip_4, ankle_4, hip_1, ankle_1, hip_2, ankle_2, hip_3,
    ankle_3), and the qpos/qvel layouts have their own offsets, so all three
    index sets are read out of the compiled model by joint name instead of being
    hardcoded.

    locked_angles is the midpoint of each joint's own range, which must not be
    hardcoded either: ankles 1 and 4 have range [+30, +70] deg while ankles 2
    and 3 have [-70, -30] deg, so a single fixed angle would pin half the legs
    outside their limits and cripple the ant far beyond losing the leg.
    """
    import mujoco

    joint_names = [f'hip_{leg_idx + 1}', f'ankle_{leg_idx + 1}']
    joint_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)
                 for n in joint_names]
    if any(jid < 0 for jid in joint_ids):
        raise ValueError(f"Ant model has no joints named {joint_names}")

    qpos_indices = [int(mj_model.jnt_qposadr[jid]) for jid in joint_ids]
    qvel_indices = [int(mj_model.jnt_dofadr[jid]) for jid in joint_ids]

    # An actuator drives this leg if its transmission targets one of its joints.
    action_indices = [i for i in range(mj_model.nu)
                      if int(mj_model.actuator_trnid[i, 0]) in joint_ids]
    if len(action_indices) != len(joint_ids):
        raise ValueError(
            f"Expected {len(joint_ids)} actuators for leg {leg_idx + 1}, "
            f"found {action_indices}")

    locked_angles = [float(mj_model.jnt_range[jid].mean()) for jid in joint_ids]

    return action_indices, qpos_indices, qvel_indices, locked_angles



class LegDamageWrapper:
    """Wrapper that disables one ant leg: zero torque + joints locked in place.

    Mirrors the Go1 quadruped wrapper in
    the paper's quadruped trainer: the damaged leg's actions are
    zeroed and its joints are pinned to a fixed angle with zero velocity, so the
    limb is frozen rather than merely limp.

    Everything the trainers need off the env other than `reset`/`step` --
    `sys`, `action_size`, `unwrapped` -- is forwarded, so `AntFeet.from_env` and
    brax's renderer work on a wrapped env unchanged.
    """

    def __init__(self, env, damaged_leg_idx):
        """
        Args:
            env: The base (already episode-wrapped) brax environment
            damaged_leg_idx: Leg to damage (0..3), or None for healthy
        """
        self._env = env
        self._damaged_leg = damaged_leg_idx

        self._action_mask = jnp.ones(env.action_size)
        if damaged_leg_idx is not None:
            mj_model = env.unwrapped.sys.mj_model
            action_idx, qpos_idx, qvel_idx, locked_angles = _leg_indices(
                mj_model, damaged_leg_idx)
            self._action_mask = self._action_mask.at[jnp.array(action_idx)].set(0.0)
            self._qpos_indices = jnp.array(qpos_idx)
            self._qvel_indices = jnp.array(qvel_idx)
            self._locked_angles = jnp.array(locked_angles)
        else:
            self._qpos_indices = None
            self._qvel_indices = None
            self._locked_angles = None

    def __getattr__(self, name):
        """Forward all other attribute access to the wrapped environment."""
        return getattr(self._env, name)

    def _lock_leg_joints(self, state):
        """Pin the damaged leg's joints to a fixed angle with zero velocity."""
        if self._damaged_leg is None:
            return state

        pipeline_state = state.pipeline_state
        q = pipeline_state.q.at[self._qpos_indices].set(self._locked_angles)
        qd = pipeline_state.qd.at[self._qvel_indices].set(0.0)
        updates = {'q': q, 'qd': qd}
        # On the mjx backend `q`/`qd` are not the state the physics reads: brax's
        # mjx pipeline steps `mjx.step`, which reads `qpos`/`qvel`, and then
        # *derives* q/qd from them (brax/mjx/pipeline.py: `q, qd = data.qpos,
        # data.qvel`). Pinning q alone would fix the observation and leave the
        # simulated leg free -- damage that reads as applied and is not. The
        # indices come out of mj_model.jnt_qposadr/jnt_dofadr, which are the
        # native MuJoCo layouts, so they address qpos/qvel directly.
        if hasattr(pipeline_state, 'qpos'):
            updates['qpos'] = pipeline_state.qpos.at[self._qpos_indices].set(
                self._locked_angles)
            updates['qvel'] = pipeline_state.qvel.at[self._qvel_indices].set(0.0)
        pipeline_state = pipeline_state.replace(**updates)
        # Recompute the observation, otherwise it would still report the joint
        # angles the sim produced just before they were pinned.
        obs = self._env.unwrapped._get_obs(pipeline_state)
        return state.replace(pipeline_state=pipeline_state, obs=obs)

    def step(self, state, action):
        masked_action = action * self._action_mask
        next_state = self._env.step(state, masked_action)
        return self._lock_leg_joints(next_state)

    def reset(self, rng):
        state = self._env.reset(rng)
        return self._lock_leg_joints(state)



def gravity_cycle(num_tasks, default_mult=DEFAULT_GRAVITY_MULT,
                  mid_mult=DEFAULT_GRAVITY_MID, high_mult=DEFAULT_GRAVITY_HIGH):
    """The gravity multiplier of each sub-task, cycling default -> mid -> high.

    Period 3, the same as `friction_cycle`. The two are therefore locked in
    phase if both axes run at once -- every x0.2 sub-task would also be a heavy
    one, and their effects could not be separated. Run one axis or the other, or
    give them different periods; the launcher scripts run gravity with
    `--friction_order none` for exactly this reason.
    """
    mults = [float(default_mult), float(mid_mult), float(high_mult)]
    return [mults[i % len(mults)] for i in range(int(num_tasks))]



def create_env_with_damaged_leg(env_name, leg_idx, episode_length=1000,
                                friction_mult=DEFAULT_FRICTION_MULT,
                                target_speed=None, speed_margin=None,
                                speed_weight=None, wrap=True,
                                backend=DEFAULT_BACKEND,
                                gravity_mult=DEFAULT_GRAVITY_MULT,
                                flipped_leg=None,
                                obs_noise_sigma=0.0, obs_noise_seed=0,
                                task_idx=0, obs_task_period=0):
    """Load `env_name` with `leg_idx` disabled and friction scaled by `friction_mult`.

    The base env is built exactly as source/studies/brax/train_GA_ant.py's `create_env`
    builds it -- episode-limited with auto-reset -- so a sub-task differs from
    the noncontinual control only in the damage and the ground. Auto-reset
    matters even though only the first episode is scored: without it the rollout
    keeps integrating physics on an already-terminated (fallen) ant and returns
    NaN rewards.

    The multiplier scales all three friction components (sliding, torsional,
    rolling) of every geom, matching what the cheetah block does
    to the cheetah. It is applied to the freshly built System, so it is applied
    exactly once -- see the module docstring on why that and `sys.geom_friction`
    rather than `sys.mj_model.geom_friction` are both load-bearing, on either
    backend.
    """
    if env_name != 'ant':
        raise ValueError(
            f"The leg-damage protocol resolves hip_N/ankle_N by name and is "
            f"specific to brax's ant xml; got env {env_name!r}.")
    # `wrap=False` for the RL trainer, which runs the env through brax's
    # training.wrap itself. With wrap=True on both sides the stack ends up
    # carrying TWO EpisodeWrappers and TWO AutoResetWrappers, and brax's
    # Evaluator then averages its episode metrics over twice the true episode
    # count -- every reported RL return came out at exactly HALF the real one.
    # Measured on the sub-task 0 policy: 32/32 episodes ran the full 1000 steps
    # for a true return of 5966, while the training log reported 2968. The NE
    # trainers roll out through their own scoring function and were never
    # affected, so the bug silently halved RL relative to NE in every ant
    # comparison.
    if wrap:
        env = envs.create(env_name, episode_length=episode_length, auto_reset=True,
                          backend=backend)
    else:
        env = envs.create(env_name, episode_length=None, auto_reset=False,
                          backend=backend)
    if friction_mult != 1.0:
        sys = env.unwrapped.sys
        env.unwrapped.sys = sys.replace(
            geom_friction=sys.geom_friction * float(friction_mult))
    if gravity_mult != 1.0:
        sys = env.unwrapped.sys
        # `sys.opt.gravity`, NOT `sys.gravity`. Both exist on a brax System and
        # both read 9.81, but only opt.gravity is the one mjx.step integrates:
        # over an 80x sweep (x0.1 to x8.0) writing sys.gravity left the torso
        # height at 0.5446-0.5455 and the return at 118.4-118.5, i.e. flat to
        # four decimals, while writing sys.opt.gravity moved both monotonically.
        # Writing the wrong one scales gravity in every readout and leaves the
        # simulation at 9.81 -- the same silent failure mode as q vs qpos above.
        env.unwrapped.sys = sys.replace(
            opt=sys.opt.replace(gravity=sys.opt.gravity * float(gravity_mult)))
    env = LegDamageWrapper(env, leg_idx)
    # Outside the damage wrapper, so a leg that is both damaged and flipped
    # stays damaged -- see MotorFlipWrapper.
    if flipped_leg is not None:
        env = MotorFlipWrapper(env, flipped_leg)
    # Outside the damage wrapper: the damage wrapper recomputes the observation
    # after pinning the joints, and the reward correction has to read the
    # velocity of the state that actually results from that.
    if target_speed is not None:
        env = TargetSpeedWrapper(env, target_speed, speed_margin, speed_weight)
    # Outermost, so the offset lands on the final observation.
    if obs_noise_sigma and float(obs_noise_sigma) > 0.0:
        # obs_task_period repeats the first `period` offsets for the rest of the
        # run, so a sub-task is REVISITED rather than replaced by a fresh draw.
        # This is the only reason forgetting is measurable on the obs-noise
        # sequence: the offset is drawn per (seed, task_idx), so without it
        # every sub-task is a new environment and the only reading available is
        # the final agent's return on an earlier one -- which conflates
        # capability lost with one unconditioned policy being asked to satisfy
        # every offset at once.
        #
        # Same semantics as cycle_task_sequence() in
        # source/studies/gymnax/continual_common.py, deliberately: index modulo period,
        # so with num_tasks=20 and period=10 sub-task 10 is bit-for-bit
        # sub-task 0 -- the unperturbed baseline -- and 11 is 1, and so on.
        env = ObsOffsetWrapper(env, obs_noise_sigma, obs_noise_seed,
                               effective_task_idx(task_idx, obs_task_period))
    return env



class MotorFlipWrapper:
    """Wrapper that inverts the sign of one leg's actuators.

    The anti-optimality mechanism for a radially symmetric body. Direction
    reversal is a no-op on the ant (it walks -x as easily as +x) and physics
    multipliers leave the carried gait a decent warm start -- but with one
    leg's motor polarity flipped, the carried policy's commands to that leg
    actively fight it: every learned reflex on that limb pushes the wrong way,
    and no re-orientation of the body can route around its own motor wiring.

    Implemented by negating the leg's action dimensions rather than by editing
    `actuator_gear` in the model: for ant's symmetric torque motors
    (ctrlrange [-1, 1]) the two are mathematically identical, and an action
    mask composes with LegDamageWrapper and both backends with no model
    surgery. Applied OUTSIDE LegDamageWrapper, so a leg that is both damaged
    and flipped stays damaged (the damage mask zeroes whatever sign this
    wrapper produces).

    `flipped_leg_idx=None` is the identity, so every existing sequence is
    reproduced bit for bit when the flip axis is off.
    """

    def __init__(self, env, flipped_leg_idx):
        self._env = env
        self._flipped_leg = flipped_leg_idx
        mask = jnp.ones(env.action_size)
        if flipped_leg_idx is not None:
            mj_model = env.unwrapped.sys.mj_model
            action_idx, _, _, _ = _leg_indices(mj_model, flipped_leg_idx)
            mask = mask.at[jnp.array(action_idx)].set(-1.0)
        self._flip_mask = mask

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, rng):
        return self._env.reset(rng)

    def step(self, state, action):
        return self._env.step(state, action * self._flip_mask)



def flip_cycle(num_tasks, num_legs=NUM_LEGS):
    """The flipped leg of each sub-task: none, leg1, none, leg2, none, ...

    Period 2*num_legs. Every second sub-task is the UNPERTURBED ant, so the
    baseline task is revisited after each flip -- which is what makes both
    halves of the comparison readable: the flip windows measure adaptation to
    an anti-optimal motor map, and the interleaved default windows measure
    what the adaptation cost in retention. Deterministic and identical for
    every method and trial, like every other cycle in this module.
    """
    seq = []
    for i in range(int(num_tasks)):
        seq.append(None if i % 2 == 0 else (i // 2) % int(num_legs))
    return seq



def leg_cycle(num_tasks, num_legs=NUM_LEGS):
    """The damaged leg of each sub-task, cycling leg 1 -> 2 -> 3 -> 4 -> 1 ...

    Deterministic and identical for every method, trial and seed; see the module
    docstring on why that is the point.
    """
    return [i % int(num_legs) for i in range(int(num_tasks))]



def friction_cycle(num_tasks, default_mult=DEFAULT_FRICTION_MULT,
                   low_mult=DEFAULT_LOW_MULT, high_mult=DEFAULT_HIGH_MULT):
    """The friction multiplier of each sub-task, cycling default -> low -> high.

    Period 3 against `leg_cycle`'s period 4: the two are coprime, so at the
    default 12 sub-tasks every (leg, multiplier) pair is visited exactly once.
    Sub-task 0 is the default multiplier, so the ground of the first sub-task is
    the noncontinual block's ground.
    """
    cycle = [float(default_mult), float(low_mult), float(high_mult)]
    return [cycle[i % len(cycle)] for i in range(int(num_tasks))]



def random_friction_sequence(rng_key, num_tasks, low_mult=DEFAULT_LOW_MULT,
                             high_mult=DEFAULT_HIGH_MULT,
                             default_mult=DEFAULT_FRICTION_MULT):
    """Log-uniformly sampled friction multiplier per sub-task.

    The Slippery-Ant protocol (Dohare et al., Nature 2024): every sub-task's
    ground is a fresh sample rather than a revisit of three fixed values. With
    3 recurring multipliers a revisit is a task the weights have already
    covered, and the learner can settle into one compromise policy for the
    whole cycle -- which protects exactly the plasticity this sequence exists
    to measure.

    Log-uniform rather than uniform because the multiplier acts as a scale:
    x0.2 and x5.0 are equally far from x1.0, while uniform sampling on
    [0.2, 5.0] would make four in five sub-tasks stickier than default.

    Sub-task 0 is pinned to `default_mult`, keeping this module's convention
    that the run starts on the unperturbed ground of the noncontinual control.

    Cost worth knowing: every distinct multiplier is baked into the compiled
    rollout as a constant, so unlike the 3-value cycle nothing here hits the
    JAX compile cache -- a 75-sub-task run pays ~75 compiles, not 3.
    """
    lo, hi = jnp.log(low_mult), jnp.log(high_mult)
    mults = jnp.exp(jax.random.uniform(
        rng_key, (int(num_tasks),), minval=lo, maxval=hi))
    sequence = [float(m) for m in mults]
    if sequence:
        sequence[0] = float(default_mult)
    return sequence



def random_leg_sequence(rng_key, num_tasks, avoid_consecutive=True):
    """Sampled sequence of damaged legs, one per sub-task.

    The pre-`--leg_order` behaviour, kept so the runs made with it can be
    reproduced. `avoid_consecutive` stops a "task switch" from being a no-op,
    but the legs are still unbalanced over a finite sequence and two methods
    only share a sequence if they share a seed.
    """
    sequence = []
    for _ in range(int(num_tasks)):
        rng_key, subkey = jax.random.split(rng_key)
        if avoid_consecutive and sequence:
            available = [leg for leg in range(NUM_LEGS) if leg != sequence[-1]]
            idx = int(jax.random.randint(subkey, (), 0, len(available)))
            leg = available[idx]
        else:
            leg = int(jax.random.randint(subkey, (), 0, NUM_LEGS))
        sequence.append(leg)
    return sequence



def create_env(env_name, episode_length, backend=DEFAULT_BACKEND,
               healthy_reward=None, target_speed=None, speed_margin=None,
               speed_weight=None):
    """Brax env for evolution: episode-limited, with auto-reset.

    `target_speed` PUTS THIS BLOCK ON THE CONTINUAL BLOCK'S OBJECTIVE, and
    without it the two are not comparable at all. The continual ant runs a
    speed-tracking reward (`continual_ant_friction_t24` pins target 2.0 for
    every sub-task); this block could only ever run brax's default unbounded
    forward-velocity reward, because nothing here took a speed argument. Two
    different objectives, two reward scales, and a "control" whose returns
    cannot be read against the runs it is the control for -- 1833.5 against
    1644.9 on GA, which is a comparison of reward functions.

    None reproduces every existing noncontinual ant run.

    Auto-reset matters even though we only score the first episode: without it,
    the rollout keeps integrating physics on an already-terminated (fallen) ant
    for the remaining steps, which diverges and returns NaN rewards. This is
    what brax's own Evaluator does -- reset the sim on `done`, and mask out
    everything after the first termination when accumulating return.

    `backend` defaults to the same value the continual block uses. This block is
    that block's control, so the two have to be the same simulator or the
    comparison is between physics engines rather than between settings; see
    "Which backend" in this module's docstring.
    """
    # healthy_reward is brax ant's +1-per-surviving-step bonus, and on mjx it is
    # the reason evolution stalls: an upright policy that never moves collects
    # ~1000 of it over a 1000-step episode, and a RANDOM policy already collects
    # 741 at generation 0. Measured on a trapped genome -- 999 steps alive,
    # 0.00 m displacement, forward reward NEGATIVE. dm_control's CheetahRun has
    # no such term (its reward is tolerance(speed) alone, so standing still
    # scores ~0), which is why the cheetah blocks never hit this and the ant
    # does. Lowering it makes the ant's reward structurally like the cheetah's.
    # None leaves brax's default (1.0) and reproduces every existing ant run.
    if target_speed is not None:
        # Delegate to the continual block's own factory with the sequence
        # switched off -- healthy legs, default friction, default gravity --
        # so the reward wrapper is literally the same code path the continual
        # runs use rather than a second implementation of it.
        return create_env_with_damaged_leg(
            env_name, leg_idx=None, episode_length=episode_length,
            target_speed=float(target_speed), speed_margin=speed_margin,
            speed_weight=speed_weight, backend=backend)

    kwargs = {} if healthy_reward is None else {'healthy_reward': float(healthy_reward)}
    return envs.create(env_name, episode_length=episode_length, auto_reset=True,
                       backend=backend, **kwargs)