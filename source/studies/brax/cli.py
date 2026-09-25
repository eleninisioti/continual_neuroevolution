"""Flags, labels and footage for the continual ant trainers.

The other half of what was ``source/studies/brax/continual_legs_common.py``. What a
sub-task *is* moved to ``source/envs/brax_ant.py``; what stayed here is
everything that only makes sense once there is a command line: the shared
argument group, the ``*_from_args`` readers that turn those flags into a
sub-task sequence, the labels that name a sub-task in a log, and the GIF
rendering.

The split is the rule in ``source/envs/__init__.py``: a study imports an
environment, never a trainer's parser. Only the ant trainers import this.
"""

from __future__ import annotations

import functools
import os

import imageio
import jax
import jax.numpy as jnp
from brax import envs

from source.utils.task_sequence import cycle_task_sequence
from source.envs.brax_ant import DEFAULT_FRICTION_MULT, DEFAULT_GRAVITY_HIGH, DEFAULT_GRAVITY_MID, DEFAULT_GRAVITY_MULT, DEFAULT_HIGH_MULT, DEFAULT_LOW_MULT, LEG_NAMES, LegDamageWrapper, flip_cycle, friction_cycle, gravity_cycle, leg_cycle, random_friction_sequence, random_leg_sequence
from source.envs.brax_common import DEFAULT_BACKEND, DEFAULT_SPEED_TARGETS, DEFAULT_SPEED_WEIGHT, TargetSpeedWrapper, speed_cycle


def flips_from_args(args):
    """The motor-flip sequence implied by `add_continual_args` flags."""
    if getattr(args, 'flip_order', 'none') == 'none':
        return [None] * int(args.num_tasks)
    return flip_cycle(args.num_tasks)


def add_continual_args(parser):
    """Sub-task flags shared by the GA/ES/DNS/RL continual ant trainers."""
    parser.add_argument('--backend', type=str, default=DEFAULT_BACKEND,
                        choices=['mjx', 'generalized', 'spring', 'positional'],
                        help="Physics backend. 'mjx' is the default and is what "
                             "the cheetah block runs on; 'generalized' is brax's "
                             "own default and reproduces every ant tree dated on "
                             "or before 2026-07-30. Not comparable across runs -- "
                             "see the module docstring.")
    parser.add_argument('--num_tasks', type=int, default=12,
                        help='Number of sub-tasks in the sequence. The default is '
                             'three full cycles of the four legs, so each is '
                             'damaged the same number of times.')
    parser.add_argument('--leg_order', type=str, default='cycle',
                        choices=['cycle', 'random', 'none'],
                        help="cycle: legs 1,2,3,4,1,... identical for every method "
                             "and trial (the sweep's setting). random: sampled from "
                             "the seed, the pre-2026-07-29 behaviour.")
    parser.add_argument('--allow_consecutive_legs', action='store_true', default=False,
                        help='--leg_order random only: permit the same leg twice in '
                             'a row, which makes a switch a no-op')
    parser.add_argument('--friction_order', type=str, default='cycle',
                        choices=['cycle', 'random', 'none'],
                        help="cycle: ground friction cycles default,low,high "
                             "alongside the leg cycle (the sweep's setting). "
                             "random: a fresh log-uniform sample in [low, high] "
                             "per sub-task, seeded off --seed so every method at "
                             "a trial shares the sequence; sub-task 0 stays at "
                             "the default (the Slippery-Ant protocol). none: "
                             "every sub-task runs at --friction_default_mult, which "
                             "reproduces the damage-only sequence used before "
                             "2026-07-29.")
    parser.add_argument('--friction_default_mult', type=float,
                        default=DEFAULT_FRICTION_MULT,
                        help='Unscaled ground; the value the noncontinual block runs at')
    parser.add_argument('--friction_low_mult', type=float, default=DEFAULT_LOW_MULT,
                        help='Slippery')
    parser.add_argument('--friction_high_mult', type=float, default=DEFAULT_HIGH_MULT,
                        help="Sticky. Values >=5 make brax's generalized solver go "
                             'non-finite on ant -- see the module docstring.')
    # The gravity axis. Off by default, so adding it changes no existing block.
    parser.add_argument('--gravity_order', type=str, default='none',
                        choices=['cycle', 'none'],
                        help="cycle: gravity cycles default,mid,high alongside "
                             "the leg cycle. none: every sub-task runs at "
                             "--gravity_default_mult. Gravity has the same "
                             "period as friction, so running both cycles locks "
                             "them in phase -- use --friction_order none with it.")
    parser.add_argument('--gravity_default_mult', type=float,
                        default=DEFAULT_GRAVITY_MULT,
                        help="Earth gravity; the value every other block runs at")
    parser.add_argument('--gravity_mid_mult', type=float, default=DEFAULT_GRAVITY_MID,
                        help='Heavy')
    parser.add_argument('--gravity_high_mult', type=float, default=DEFAULT_GRAVITY_HIGH,
                        help="Very heavy. Multipliers BELOW 1.0 float the torso "
                             "into ant's healthy_z_range bound and terminate the "
                             "episode early -- see the module constants.")
    parser.add_argument('--obs_noise_range', type=float, default=0.0,
                        help='Sigma of the fixed per-sub-task observation-offset '
                             'vector, the gymnax continual protocol on the ant. '
                             '0 (default) disables it and reproduces every '
                             'existing sequence; sub-task 0 is always '
                             'unperturbed. Offsets are seeded off --seed and the '
                             'sub-task index alone, so every method at a trial '
                             'faces the same vectors.')
    parser.add_argument('--task_period', type=int, default=0,
                        help='Revisit observation offsets with this period: the '
                             'first task_period offsets repeat for the rest of '
                             'the run, so each sub-task is seen more than once '
                             'and forgetting can be read as "what does this '
                             'sub-task score now vs when its own interval '
                             'ended". Without it the offset is a fresh draw '
                             'every sub-task and nothing is ever revisited. '
                             'Same meaning as --task_period in the gymnax '
                             'continual trainers; 0 (default) reproduces every '
                             'existing ant tree. Affects only the offset -- the '
                             'leg, friction, gravity and speed sequences are '
                             'already cycles and revisit on their own.')
    parser.add_argument('--flip_order', type=str, default='none',
                        choices=['cycle', 'none'],
                        help="cycle: every odd sub-task inverts one leg's motor "
                             "polarity (none, leg1, none, leg2, ...), making the "
                             "carried policy anti-optimal on that limb; the "
                             "even sub-tasks are the unperturbed ant. none: no "
                             "flips, which reproduces every existing sequence.")
    # The speed axis is independent of the friction axis, so a block can run
    # either or both: friction_order=cycle + speed_order=none is the friction
    # sequence, the reverse is the target-speed sequence. Default off, so adding
    # this flag does not change what the friction runs do.
    parser.add_argument('--speed_order', type=str, default='none',
                        choices=['cycle', 'none'],
                        help='cycle: the reward tracks a target speed that cycles '
                             'through --speed_targets alongside the leg cycle. '
                             'none: the default brax ant reward, unbounded in '
                             'forward speed.')
    parser.add_argument('--speed_targets', type=str,
                        default=",".join(f"{t:g}" for t in DEFAULT_SPEED_TARGETS),
                        help='Comma-separated target speeds to cycle through')
    parser.add_argument('--speed_margin', type=float, default=None,
                        help='Gaussian sigma of the speed-tracking credit, absolute. '
                             'Default None = DEFAULT_SPEED_MARGIN_RATIO x target, '
                             'which keeps both sub-tasks equally forgiving.')
    parser.add_argument('--speed_weight', type=float, default=DEFAULT_SPEED_WEIGHT,
                        help='Weight on the tracking term relative to the 1.0/step '
                             'survival bonus. Too low and standing still scores '
                             'almost as well as tracking.')
    return parser


def _fold_by_task_period(fn):
    """Make a `*_from_args` sequence builder honour `--task_period`.

    THE REVISIT IS THE POINT OF THE CONTINUAL PROTOCOL, and until this existed
    it applied to the observation-noise offsets and nothing else: `task_period`
    reached `effective_task_idx` inside the noise path only, so a run asking for
    `--task_period 12` on the friction sequence got 24 distinct frictions and no
    revisit at all, silently. Without a revisit forgetting cannot be measured --
    see `cycle_task_sequence` in source/utils/task_sequence.py for why a
    final-agent score on an earlier sub-task is not the same question.

    Applied to every axis rather than to friction alone, so a flag that claims
    to fold the sequence folds all of it.
    """
    @functools.wraps(fn)
    def wrapper(args, *rest, **kw):
        sequence = fn(args, *rest, **kw)
        period = int(getattr(args, 'task_period', 0) or 0)
        return cycle_task_sequence(sequence, period)
    return wrapper


@_fold_by_task_period
def legs_from_args(args, seed=None):
    """The sub-task sequence implied by the parsed `add_continual_args` flags."""
    if args.leg_order == 'none':
        # Healthy ant for every sub-task. Needed by the target-speed sequence:
        # with two speeds the speed cycle has period 2, which shares a factor
        # with the 4-leg cycle, so leg and speed would be locked in phase (leg 1
        # always slow, leg 2 always fast) and their effects could not be
        # separated. Dropping the damage removes the confound and matches
        # C-CHAIN's Continual Quadruped, which perturbs speed on one body.
        return [None] * int(args.num_tasks)
    if args.leg_order == 'cycle':
        return leg_cycle(args.num_tasks)
    key = jax.random.key(seed if seed is not None else args.seed)
    return random_leg_sequence(key, args.num_tasks,
                               avoid_consecutive=not args.allow_consecutive_legs)


@_fold_by_task_period
def frictions_from_args(args, seed=None):
    """The friction multiplier sequence implied by `add_continual_args` flags.

    `seed` matters only for --friction_order random and should be the run's
    --seed, exactly as legs_from_args takes it: the sweep gives every method
    the same seed at a given trial, which is what makes the sampled sequence
    shared across methods -- the guarantee the cycle gets for free. The key is
    folded to its own stream so the friction draws stay independent of the leg
    draws at the same seed.
    """
    order = getattr(args, 'friction_order', 'cycle')
    if order == 'none':
        return [float(args.friction_default_mult)] * int(args.num_tasks)
    if order == 'random':
        key = jax.random.fold_in(
            jax.random.key(seed if seed is not None else args.seed), 1)
        return random_friction_sequence(key, args.num_tasks,
                                        args.friction_low_mult,
                                        args.friction_high_mult,
                                        args.friction_default_mult)
    return friction_cycle(args.num_tasks, args.friction_default_mult,
                          args.friction_low_mult, args.friction_high_mult)


@_fold_by_task_period
def gravities_from_args(args):
    """The gravity multiplier sequence implied by `add_continual_args` flags."""
    if getattr(args, 'gravity_order', 'none') == 'none':
        return [float(getattr(args, 'gravity_default_mult', DEFAULT_GRAVITY_MULT))] \
            * int(args.num_tasks)
    return gravity_cycle(args.num_tasks, args.gravity_default_mult,
                         args.gravity_mid_mult, args.gravity_high_mult)


@_fold_by_task_period
def speeds_from_args(args):
    """The target-speed sequence implied by `add_continual_args` flags.

    A list of None (rather than an empty list) when the axis is off, so callers
    zip it with the leg sequence unconditionally instead of branching.
    """
    if getattr(args, 'speed_order', 'none') == 'none':
        return [None] * int(args.num_tasks)
    targets = [float(x) for x in str(args.speed_targets).split(',') if x.strip()]
    if not targets:
        raise ValueError("--speed_order cycle needs at least one --speed_targets")
    return speed_cycle(args.num_tasks, targets)


def task_label(leg_idx, friction_mult=None, target_speed=None,
               gravity_mult=None):
    """Filesystem-safe label for a sub-task, e.g. 'leg1_fric0p20'.

    `friction_mult=None` gives the bare leg label the damage-only sequence wrote,
    so runs made before friction was added keep their directory names. Gravity
    is omitted at x1.0 for the same reason: a friction or speed block's
    directory names are unchanged by the axis existing.
    """
    label = LEG_NAMES[leg_idx].lower()
    if friction_mult is not None:
        label = f"{label}_fric{float(friction_mult):.2f}"
    if target_speed is not None:
        label = f"{label}_spd{float(target_speed):.2f}"
    if gravity_mult is not None and float(gravity_mult) != 1.0:
        label = f"{label}_grav{float(gravity_mult):.2f}"
    return label.replace(".", "p")


def add_gif_args(parser):
    """`--gifs_per_task`, matching the cheetah continual trainers' flag of the name."""
    parser.add_argument('--gifs_per_task', type=int, default=3,
                        help='Rollouts of the sub-task best genome rendered to GIFs at '
                             'each sub-task boundary. 0 skips rendering entirely')
    return parser


def save_rl_task_gifs(env_name, leg_idx, friction_mult, target_speed,
                      normalizer_params, policy_params,
                      policy_hidden_sizes, value_hidden_sizes,
                      output_dir, task_idx, episode_length, key,
                      num_gifs=3, frame_stride=4, speed_margin=None,
                      speed_weight=None, backend=DEFAULT_BACKEND,
                      gravity_mult=DEFAULT_GRAVITY_MULT,
                      activation=None):
    """Render the RL policy at a sub-task boundary, one GIF per rollout.

    The RL counterpart of `save_task_gifs`, which cannot be reused: the NE
    version renders a flat genome through a scoring function that was already
    JIT-compiled against the env, whereas the RL policy is a brax PPO network
    that has to be rebuilt from its layer sizes and driven with its own
    normalizer state.

    Two details are load-bearing and were both got wrong the first time they
    were written by hand elsewhere in this repo:

      * `preprocess_observations_fn=running_statistics.normalize`. The trainer
        runs with normalize_observations=True and builds its networks that way,
        so a network built with the default identity preprocessor is handed
        normalizer params it never applies and sees raw observations. The
        resulting policy flails: an evaluation harness that made this mistake
        reported v_x ~0 and a return of -137 for weights that had earned 10103.
      * `activation`. Same failure mode as the preprocessor: the rebuilt network
        has to use the trainer's hidden activation, not make_ppo_networks'
        default swish, or the GIF shows tanh weights driven through a swish
        network. Defaults to tanh, which is what the ant trainers run
        (source/algorithms/networks.py POLICY_ARCH['brax']).
      * `auto_reset=False`, and the rollout stops at `done`. With auto-reset the
        env teleports a fallen ant back to the origin and keeps going, which
        both produces a GIF that cuts discontinuously mid-clip and inflates any
        reward summed over it.

    Rendering is best-effort: a failure here costs footage, not the run, and the
    checkpoint has already been written by the time this is called.
    """
    if num_gifs <= 0 or policy_params is None:
        return 0

    import os

    import imageio
    import jax
    import jax.numpy as jnp
    import numpy as np
    from brax import envs
    from brax.io import image as brax_image
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    gifs_dir = os.path.join(output_dir, "gifs",
                            f"task_{task_idx:02d}_"
                            f"{task_label(leg_idx, friction_mult, target_speed, gravity_mult)}")
    written = 0
    try:
        os.makedirs(gifs_dir, exist_ok=True)

        # Rebuilt rather than reusing the training env: that one is vmapped and
        # auto-resetting, and a single clean episode is what renders.
        # Same backend as training, or the clip shows a policy walking on
        # physics it was never trained against.
        base = envs.create(env_name, episode_length=episode_length,
                           auto_reset=False, backend=backend)
        if gravity_mult is not None and float(gravity_mult) != 1.0:
            g = base.unwrapped.sys
            base.unwrapped.sys = g.replace(
                opt=g.opt.replace(gravity=g.opt.gravity * float(gravity_mult)))
        if friction_mult is not None and friction_mult != 1.0:
            sys = base.unwrapped.sys
            base.unwrapped.sys = sys.replace(
                geom_friction=sys.geom_friction * float(friction_mult))
        env = LegDamageWrapper(base, leg_idx)
        if target_speed is not None:
            env = TargetSpeedWrapper(env, target_speed, speed_margin, speed_weight)

        obs_size = int(jax.jit(env.reset)(jax.random.key(0)).obs.shape[-1])
        network = ppo_networks.make_ppo_networks(
            observation_size=obs_size,
            action_size=env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
            policy_hidden_layer_sizes=tuple(policy_hidden_sizes),
            value_hidden_layer_sizes=tuple(value_hidden_sizes),
            activation=jax.nn.tanh if activation is None else activation,
        )
        policy = ppo_networks.make_inference_fn(network)(
            (normalizer_params, policy_params), deterministic=True)

        reset = jax.jit(env.reset)

        # Rolled out with lax.scan, not a Python loop. A loop that tests
        # `bool(state.done)` each step forces a device->host sync 1000 times per
        # GIF; inline at every sub-task boundary that took over ten minutes and
        # stalled training. The whole episode is scanned on device and trimmed
        # afterwards, which is also how the NE side does it.
        @jax.jit
        def rollout(rng):
            init = reset(rng)

            def body(carry, step_key):
                state, done_before = carry
                action, _ = policy(state.obs, step_key)
                nxt = env.step(state, action)
                # Freeze once terminated so the trimmed tail is a still frame
                # rather than physics continuing on a fallen ant.
                state = jax.tree.map(
                    lambda a, b: jnp.where(done_before, a, b), state, nxt)
                done = jnp.logical_or(done_before, nxt.done > 0.5)
                reward = jnp.where(done_before, 0.0, nxt.reward)
                return (state, done), (state.pipeline_state, reward, done)

            keys = jax.random.split(rng, episode_length)
            _, (states, rewards, dones) = jax.lax.scan(
                body, (init, jnp.array(False)), keys)
            return init.pipeline_state, states, rewards, dones

        for gif_idx in range(num_gifs):
            key, roll_key = jax.random.split(key)
            first, traj, rewards, dones = rollout(roll_key)
            dones = np.asarray(dones)
            alive = int(np.argmax(dones)) + 1 if dones.any() else episode_length
            total = float(np.asarray(rewards)[:alive].sum())

            frames = [first] + [jax.tree.map(lambda x: x[i], traj)
                                for i in range(0, alive, frame_stride)]
            images = brax_image.render_array(env.sys, frames, height=240, width=320)
            path = os.path.join(
                gifs_dir,
                f"trajectory_{gif_idx:02d}_reward{total:.0f}_steps{alive}.gif")
            imageio.mimsave(path, images, fps=30, loop=0)
            written += 1
        print(f"  Saved {written} GIFs for sub-task {task_idx} -> {gifs_dir}",
              flush=True)
    except Exception as exc:  # noqa: BLE001 - footage is never worth failing a run for
        print(f"  Warning: RL GIF rendering failed for sub-task {task_idx}: {exc}",
              flush=True)
    return written

def save_task_gifs(env, rollout_with_trajectory, flat_params, output_dir,
                   task_idx, leg_idx, episode_length,
                   key, num_gifs=3, frame_stride=4, friction_mult=None,
                   target_speed=None, gravity_mult=None):
    """Render the sub-task's best genome and write one GIF per rollout.

    Layout matches the cheetah block's version --
    `gifs/task_NN_<label>/trajectory_KK_rewardR.gif` -- so an ant run and a
    cheetah run can be flipped through the same way.

    `rollout_with_trajectory` is the second return value of the noncontinual
    trainers' make_scoring_fn. It is JIT-compiled against the *current*
    sub-task's env, so this must be called before the next sub-task rebuilds it.

    Rendering is best-effort: a failure here costs footage, not the run, and the
    run has already written its checkpoint by the time this is called. Returns
    the number of GIFs actually written.
    """
    if num_gifs <= 0 or flat_params is None:
        return 0

    # Imported here rather than at module scope: this module is imported by
    # trainers that never render, and imageio pulls in a sizeable stack.
    import os

    import imageio
    from brax.io import image as brax_image

    gifs_dir = os.path.join(output_dir, "gifs",
                            f"task_{task_idx:02d}_"
                            f"{task_label(leg_idx, friction_mult, target_speed, gravity_mult)}")
    written = 0
    try:
        os.makedirs(gifs_dir, exist_ok=True)
        for gif_idx in range(num_gifs):
            key, gif_key = jax.random.split(key)
            total_reward, trajectory_states = rollout_with_trajectory(flat_params, gif_key)
            total_reward = float(total_reward)

            # Every frame_stride'th frame: rendering is the expensive part and a
            # 1000-step episode at 30fps is longer than anyone watches anyway.
            frames = [jax.tree.map(lambda x: x[i], trajectory_states)
                      for i in range(0, episode_length, frame_stride)]
            images = brax_image.render_array(env.sys, frames, height=240, width=320)
            path = os.path.join(gifs_dir,
                                f"trajectory_{gif_idx:02d}_reward{total_reward:.0f}.gif")
            imageio.mimsave(path, images, fps=30, loop=0)
            written += 1
        print(f"  Saved {written} GIFs for sub-task {task_idx} -> {gifs_dir}")
    except Exception as exc:  # noqa: BLE001 - footage is never worth failing a run for
        print(f"  Warning: GIF rendering failed for sub-task {task_idx}: {exc}")
    return written
