"""The continual protocol's task sequence, shared by every suite.

`cycle_task_sequence` lived in `source/studies/gymnax/continual_common.py` and so was
available to gymnax alone. brax grew its own partial version --
`effective_task_idx` in source/envs/brax_ant.py -- which was wired into the
observation-noise offsets and NOTHING else, so `--task_period` silently did
nothing to the friction, leg, speed or gravity sequences. A flag that appears
to work and does not is worse than a missing one, which is why this is now one
definition that every axis folds through.
"""


def cycle_task_sequence(sequence, period):
    """Repeat the first `period` sub-tasks until the sequence is filled.

    Makes the run revisit sub-tasks it has already learned, which is what lets
    forgetting be measured at all in this setup. Without a revisit the only
    reading available is the final agent's return on an earlier sub-task, and
    that conflates two things: the capability being lost, and a single
    unconditioned policy being asked to satisfy every sub-task's observation
    shift at once, which it is not free to do. On a revisit the question is
    asked the way the continual-learning literature asks it -- what does this
    sub-task score *now*, against what it scored when its own interval ended --
    and no task label is needed at any point, only the return.

    Continual World can skip this because it gives each task its own policy
    head, so the final shared body can be queried through the right head. Our
    learner has one head and no task signal. The C-CHAIN paper's Continual
    Quadruped (Walk-Run-Walk) takes the revisit route for the same reason.

    A flat repeat holds the confound still rather than removing it: every
    sub-task is revisited after exactly `period` others, so the gap is a
    constant of the design and not a per-task nuisance. Varying it on purpose
    is a different experiment -- it measures how forgetting grows with the gap,
    and needs a schedule built for that.

    `period` of 0, None, or anything at least as long as the sequence leaves it
    untouched, so the default reproduces every run already on disk.
    """
    if not period or period >= len(sequence):
        return sequence
    return [sequence[i % period] for i in range(len(sequence))]


# ---------------------------------------------------------------------------
# The PHYSICS sub-task family (`--task_type param`)
# ---------------------------------------------------------------------------
#
# The default continual gymnax sub-task is an observation offset. This is the
# other axis: the observation is untouched and a named group of the body's
# physics parameters is rescaled instead -- the pole length on CartPole, the
# link masses on Acrobot, gravity on MountainCar.
#
# ONE TABLE FOR EVERY TRAINER, WHICH IT WAS NOT BEFORE. Until 2026-09-08 each
# of GA, DNS and RL carried its own `PARAM_CONFIGS` literal, and they DID NOT
# AGREE: CartPole's gravity range was [0.98, 98.0] in the two NE trainers and
# [0.098, 198.0] in the RL one, MountainCar's [0.000833, 0.0075] against
# [0.00125, 0.005]. Both halves of the comparison drew from that range with the
# same trial-seeded key, so the sub-task sequences silently diverged between
# the NE and RL arms of what is meant to be one compute-matched experiment
# (CLAUDE.md rule c). ES/NES had no param branch at all and would have run the
# observation-noise experiment into the same tree.
#
# A SUB-TASK IS A MULTIPLIER, NOT AN ABSOLUTE VALUE. The old sequence drew the
# parameter uniformly over an absolute range, which put sub-task 0's default
# (9.8 for CartPole gravity, in [0.98, 98.0]) at an arbitrary point of the
# distribution and made the family asymmetric in a way nothing chose. Here
# sub-task 0 is multiplier 1.0 -- the stock body, identical to the
# noncontinual block, exactly as sub-task 0 of the noise family is the
# unperturbed environment -- and every later sub-task draws a multiplier
# LOG-uniformly over `mult_range`, so a factor of two up and a factor of two
# down are equally likely. That is the counterpart of the noise family's
# zero-mean Gaussian offset: symmetric about the stock body, with no direction
# built in.
#
# WHICH KNOB, AND HOW FAR, ARE MEASURED NUMBERS. They come from
# `source/envs/gymnax_classic.py`'s `ENV_CONFIGS[env]['physics']`, chosen on
# 2026-09-05 by scoring specialists trained on the stock body under every
# available knob at 0.25x-4x. Two constraints fix each entry: the stock
# specialist must NOT already solve the rescaled body (or the sub-task is not
# a sub-task), and a specialist trained on it must be able to (or the sub-task
# is unlearnable and every method floors together).
#
#   CartPole    `length` at 2x drops stock specialists from 500 to 38.6 and is
#               trainable; pole mass, cart mass and force change nothing at any
#               multiplier, and gravity only at 4x. Range 0.5x-2x.
#   Acrobot     `mass` at 1.15x drops them from -68.9 to -93.6 and a specialist
#               trained there reaches -74.9; at 1.25x it ends at -85.5, past
#               the -80 threshold, so 1.15 is the edge of learnable. Range
#               1/1.15x-1.15x.
#   MountainCar `gravity` at 1.5x drops them from -105.6 to -193.8 and a
#               specialist trained there reaches -126; 2x is out of reach.
#               Range 1/1.5x-1.5x.
#
# The ranges are therefore *tight* on Acrobot and MountainCar and wide on
# CartPole. That is a property of the bodies, not a choice: those are the
# intervals over which a rescaling is both a real shift and still solvable.
#
# One consequence to keep in mind when reading the figures: on the "easy" side
# of each range (a shorter pole, lighter links, weaker gravity) the stock
# policy transfers zero-shot, so those sub-tasks are mild. The noise family has
# the same property -- a small offset draw is a mild sub-task -- and in both
# cases the fix is that the sequence is trial-seeded and SHARED, so every
# method meets the same mild and the same hard sub-tasks in the same order.
GYMNAX_PHYSICS_TASKS = {
    'CartPole-v1':    {'param': 'length',  'mult_range': (0.5, 2.0)},
    'Acrobot-v1':     {'param': 'mass',    'mult_range': (1.0 / 1.15, 1.15)},
    'MountainCar-v0': {'param': 'gravity', 'mult_range': (1.0 / 1.5, 1.5)},
}


def physics_mult_sequence(trial, num_tasks, mult_range, period=0):
    """The per-trial multiplier sequence for `--task_type param`.

    Sub-task 0 is 1.0 -- the stock body, so it is the noncontinual experiment
    exactly as the noise family's sub-task 0 is. Later sub-tasks draw
    log-uniformly over `mult_range`.

    Seeded from `trial` alone with the same `trial * 7919` key the noise family
    uses, and deliberately NOT from the training RNG, so every method at a
    given trial faces the same bodies in the same order. `period` folds through
    `cycle_task_sequence`, which the param branch used to skip entirely: a
    20-sub-task run at period 10 was giving the noise arms each sub-task twice
    and the param arms twenty distinct ones, so the two were not measuring
    forgetting on the same schedule -- and the param arms were not measuring it
    at all, since nothing was ever revisited.
    """
    import math

    from jax import random

    lo, hi = float(mult_range[0]), float(mult_range[1])
    log_lo, log_hi = math.log(lo), math.log(hi)
    rng = random.key(int(trial) * 7919)
    mults = [1.0]
    for _ in range(1, num_tasks):
        rng, draw_key = random.split(rng)
        mults.append(float(math.exp(
            float(random.uniform(draw_key, minval=log_lo, maxval=log_hi)))))
    return cycle_task_sequence(mults, period)


def action_flip_sequence(num_tasks, period=0, env_name=None, trial=0):
    """The per-sub-task action-reversal flags for `--task_type actions`.

    Sub-task 0 is 0 -- the stock action order, so it is the noncontinual
    experiment exactly as sub-task 0 is under `noise` and `param` -- and the
    flag then ALTERNATES: 0, 1, 0, 1, ... Sub-task i's action order is
    reversed (a -> n-1-a) when its flag is 1, and `FlipEnv` in
    source/envs/gymnax_classic.py is what applies it.

    NOT drawn from the trial seed, and that is the whole difference from
    `physics_mult_sequence`. A reversal flag has two states, so a random draw
    would give runs of consecutive sub-tasks in the SAME regime -- boundaries
    at which nothing changes -- and a different number of real switches per
    trial. Alternation makes every boundary a real reversal and gives every
    trial the same number of them, which is what CLAUDE.md rule (c) asks for.
    The trial still differs in its training seed; it is the sub-task SEQUENCE
    that is shared, as it is under the other two families.

    `period` folds through `cycle_task_sequence` for consistency, though it is
    a no-op on an alternating sequence of even period: 20 sub-tasks at period
    10 already visit each regime ten times either way.

    WHY THIS IS THE HARD FAMILY. The observation is untouched, so the two
    regimes are indistinguishable to the policy and they demand OPPOSITE
    outputs at the same input. For a memoryless policy of the observation --
    the NE MLP and PPO's actor alike -- there is provably no single policy that
    scores well on both, which is exactly what the physics family turned out to
    lack: pole length admits one policy covering the whole range, so widening
    it stretched the transient without creating interference. Pass a cue (the
    `actions_cue` option of `TaskSpec`) to make the regime identifiable from
    the observation and a generalist possible again; that is a DIFFERENT and
    strictly easier experiment.
    """
    if env_name and 'DeepSea' in str(env_name):
        # DeepSea's action sub-task is a MAP, not a reversal, and there are as
        # many maps as the schedule has distinct sub-tasks: flag f > 0 seeds
        # one map (DeepSeaEnv folds it into its key), folded with the trial
        # so trials face different maps as they face different offsets under
        # `noise`. The `period` distinct sub-tasks are `period` distinct
        # maps, and the revisit brings each one back.
        distinct = period if period and period < num_tasks else num_tasks
        flags = [0] + [1000 * int(trial) + k for k in range(1, distinct)]
        return [flags[i % distinct] for i in range(num_tasks)]
    flips = [i % 2 for i in range(num_tasks)]
    return cycle_task_sequence(flips, period)
