"""What the mjx study runs: the bodies, the cells, the arms, and the budget.

This is a study, not a trainer. Neither loop it drives lives here:

    NE   ``source/studies/generalists/train_nes.py:run_nes``
    RL   ``source/studies/generalists/train_ppo.py:run_ppo``

Both are already suite-generic -- they resolve an environment through
``source/envs/registry.py`` and touch a body only through ``source/envs/mjx.py``,
``source/envs/brax_common.py`` and ``source/envs/brax_ant.py``. This is the
MiniGrid arrangement (``source/studies/minigrid/settings.py``) on the two
continuous-control bodies, and it is deliberately NOT a fork of the trainers
under ``source/studies/brax/``: those predate the centroid, have no gaussian
DNS arm and would be a second implementation of every method, which CLAUDE.md
(a) forbids. They are what produced ``runs_repro2/brax`` and nothing here reads
them. The cheetah's old trainers (``source/studies/mujoco/``) and its old
environment (``source/envs/mjx_cheetah.py``) were deleted on 2026-09-08 and do
not exist on this checkout at all.

TWO BODIES, ONE FILE, because the difference between them is a table of
constants and nothing else -- CLAUDE.md (a) and (b). Adding a third body is a
row in ``BODIES`` plus three rows in ``CELLS``.


## What is different from runs_repro2, and why these are re-runs

Three things the reported trees cannot answer, on either body:

  centroid    the old trainers never save the mean of the population's
              weights, so a curve there can only be the elite -- the same
              defect that made the gymnax centroid figure measure the wrong
              network (fixed 2026-09-09). The shared runner saves ``centroid``
              (population mean), ``popmean`` and one checkpoint per phase into
              ``checkpoints.npz``.
  the pair    the old DNS breeds with Iso+LineDD only, so a GA-vs-DNS gap
              confounds the SELECTION rule with the variation operator. The
              paper reports the gaussian column, so the NE pair here is ``ga``
              (gaussian + truncation) and ``dns_gaussian`` (gaussian +
              dominated novelty) at one mutation width.
  the curve   the old reported number is the search's own selection statistic
              and carries the winner's curse. Here every generation re-scores
              the centroid and the population mean on FRESH keys
              (``EVAL_EPISODES`` episodes), feeding nothing back.

And, on the cheetah only, a fourth: ``runs_repro2/mujoco`` ran TWELVE sub-tasks
at ``task_period 0``, so nothing was ever revisited and retention was not
measurable at all. That is the column the centroid and plasticity figures
exist to fill.


## The cheetah is a different BODY from the one repro2 ran

``runs_repro2/mujoco`` is dm_control's CheetahRun through mujoco_playground.
This is brax's ``halfcheetah`` on the same MJX physics, since 2026-09-08. Both
are 17-dim observation and 6 actuators, so the policy, the searchers and the
analysis are unchanged -- but the REWARD is not:

    repro2   dm_control ``rewards.tolerance`` at ``_RUN_SPEED`` 10, ONE-sided
             (full credit at or above the target), per-step in [0, 1], so an
             episode caps at 1000.
    here     ``TargetSpeedWrapper``: a TWO-sided Gaussian peaking AT the
             target, weight 5, margin 0.5x the target, on a body with no
             healthy bonus and no termination. Measured 2026-09-09: a
             random-action cheetah scores ~460 an episode and an untrained
             network 470-580, against a ceiling of ~5000.

So no cheetah number from before 2026-09-08 is on this scale, and matching
repro2's PPO hyperparameters would match their NAMES rather than their
meaning -- ``reward_scaling`` 0.1 against a reward that caps at 1/step is a
different object from 0.1 against one that caps at 5/step. What IS carried
over from repro2 is the per-body constants below, because those are properties
of the body and the policy parameterisation, and neither changed.

``target_speed`` 10.0 is dm_control's constant for dm_control's cheetah xml and
is PROVISIONAL on this one. It needs the treatment the ant's 2.0 got -- a
specialist run confirming the target is reachable and not trivially so -- and
``scripts/train/probe_cheetah_stationary.sh`` is that run. Do not read a
cheetah continual figure before it has passed.


## The cells

Three per body: two continual families and the stationary control.

    <body>_noise      a sub-task is a fixed offset vector added to the body's
                      observations, sigma 2.0. The BODY is unchanged.
    <body>_friction   a sub-task is a ground-friction multiplier drawn
                      log-uniform in [0.05, 5.0] per trial -- the Slippery-Ant
                      draw ``runs_repro2`` used. The PHYSICS changes and the
                      observation does not.
    <body>            stationary control, sub-task 0 throughout.

Ten sub-tasks over twenty phases, so each is visited exactly twice and the
second visit is the retention measurement -- the same structure as the gymnax
grid's ``NUM_TASKS=20 TASK_PERIOD=10``, which is what lets every body be read
side by side.

ONE stationary cell per body serves BOTH families and that is not a shortcut:
sub-task 0 is the unperturbed body under either task_mod (a zero offset, a x1.0
multiplier), the sub-task vectors are drawn off the trial rather than the run's
key stream, and a ``task0`` run therefore never touches the difference. The two
would be the same run twice, at ~1-3 h a trial. It is filed under the friction
family so its nine idle evaluation columns are zero-shot friction transfer.

The stationary cells are checkpointed on the SAME 16-generation phase grid as
the continual ones even though nothing changes at those points. It costs
nothing and it is what gives the plasticity figure a no-task-change control: a
dormancy or rank number is evidence about task changes only if the same
measurement on a stationary run of the same length does something different.


## The budget, and it is one budget for both bodies

Compute-matched in environment steps, and asserted by ``check()`` at the start
of every trial rather than asserted in this comment:

    NE   320 generations x 512 population x 3 rollouts x 1000 steps = 4.9152e8
    RL   48,000 updates x 512 envs x 20 steps                       = 4.9152e8

split into the SAME twenty phases -- 16 generations against 2400 updates -- so
both families meet a task change at the same point on that shared wall of
steps, which is CLAUDE.md (c). The RL half is ``train_ppo.PPO_CONFIGS``, which
gives both bodies the same ``_MJX_PPO`` entry; it is repeated here only as the
phase count, so the two cannot drift apart silently.

This is also repro2's own per-sub-task budget on BOTH bodies: 24,576,000
environment steps a sub-task, which is 16 generations at pop 512 x 3 evals.
Only the number of phases differs (20 here against repro2's 12 on the cheetah
and 24 on the ant), and that difference is what buys the revisit.


## Where the per-body numbers come from

Read back off this repo's own finished runs, not retyped from a paper. Sigma
does NOT transfer between bodies -- the GA mutates at 0.01 on the ant and 0.1
on the cheetah, and copying one row onto the other would be a 10x handicap
dressed up as consistency. It does transfer across a change of SIMULATOR at a
fixed body and policy, which is why the cheetah row is repro2's.


## No arm is told where a boundary is

CLAUDE.md (d). Nothing here passes a boundary signal: C-CHAIN's
``cchain_reset_on_switch`` is off, and the GA and DNS re-evaluate their stored
population every generation rather than only at a transition. ``--oracle`` on
the CLI builds the deliberately-informed control arm and is the only way to
turn any of it on.


## No solved threshold

``source/envs/mjx.py`` says it outright: neither body has one. The analysis
reports the SCORE, and nothing downstream invents a constant.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The bodies
# ---------------------------------------------------------------------------
#
# `env_name` is the suite entry in source/envs/mjx.py. Everything else is a
# constant this body's own runs were made at.
#
# `target_speed` is spelled out rather than left to `ENV_CONFIGS` because it
# changes what the returns MEAN -- brax's stock reward is unbounded forward
# velocity and the paper measured that under it a body re-routes around a
# friction change, so friction alone is not a shift. A silent default would
# show up in no log.
#
# `ne_arms` is the per-body search settings. `family` picks the runner;
# everything else is passed straight through to it. `searcher_kwargs` goes to
# `ne.build_searcher`, the one place that knows which knobs define which
# method. `sigma` is the TOP-LEVEL name, not `sigma_init` inside
# searcher_kwargs: `run_nes` forwards its own `sigma` as `sigma_init`, so
# passing both is a duplicate keyword and the run dies at construction.

# The gaussian pair's shared AURORA settings. Identical on both bodies in
# runs_repro2 -- a 1000-step episode is subsampled to 10 descriptor steps --
# so this is one constant rather than a per-body row.
_AURORA = dict(descriptor='aurora', traj_steps=10,
               aurora_latent_dim=6, aurora_train_ratio=8)


def _ne_arms(ga_sigma, es_sigma, es_lr, nes_sigma, nes_lr,
             ga_init_around_mean=True):
    """The four NE arms at one body's widths.

    The GAUSSIAN PAIR shares `ga_sigma`, so `ga` vs `dns_gaussian` isolates the
    SELECTION RULE and nothing else. That is the whole point of the pair and is
    why the mutation width is a single argument here rather than two.

    `ga_init_around_mean` is shared by the pair for the same reason. True is
    `GASearcher`'s default -- jittered copies of the flax seed policy, the
    study's "every arm starts at one point" convention. It is a per-BODY
    choice because on the ant it is not a neutral one: see BODIES['ant'].
    """
    return {
        # NES: standardized fitness, plain SGD.
        'nes': dict(method='nes', sigma=nes_sigma, learning_rate=nes_lr,
                    optimizer='sgd', shaping='zscore'),
        # OpenES: the SAME searcher with the two settings that define it --
        # centered ranks and Adam. Named `es` to match the gymnax tree.
        'es': dict(method='openes', sigma=es_sigma, learning_rate=es_lr,
                   optimizer='adam', shaping='centered_rank'),
        'ga': dict(method='ga', sigma=ga_sigma,
                   searcher_kwargs=dict(
                       elite_ratio=0.1,
                       init_around_mean=ga_init_around_mean)),
        'dns_gaussian': dict(
            method='dns_gaussian', sigma=ga_sigma,
            searcher_kwargs=dict(cross_over_rate=0.0, k=3,
                                 init_around_mean=ga_init_around_mean,
                                 **_AURORA)),
    }


BODIES = {
    'ant': dict(
        env_name='ant',
        # The paper's continual ant reward, `TargetSpeedWrapper` at 2.0 m/s.
        # Confirmed reachable by a specialist run before the block was read.
        target_speed=2.0,
        # ~2.5x the ant's own per-dim observation spread (median std 0.804,
        # re-measured 2026-09-09), and the value runs_repro2/brax was made at.
        noise_range=2.0,
        # From `projects/generalists/runs/ant_*_2task_ant` and the
        # `ant_nes_sweep_*` grid. NES lr 0.005 collapses on this body (final
        # generalist 15-455 across nine trials against 1130-1410 at 0.0025).
        # THE GAUSSIAN PAIR INITIALISES AS runs_repro2 DID: `N(0, init_scale)`
        # on every weight, no seed policy. This is not a preference, it is what
        # makes the arm run at all on this body, and it is measured twice over.
        #
        # `ne.py`'s own note (2026-09-06): with the default seed-policy init the
        # ant's actions are twice as large, every jittered copy falls within ~30
        # steps, and a GA at sigma 0.01 never finds the upright posture from
        # there -- while 4% of an N(0, 0.1) population stands for the whole
        # episode. Reproduced on CLUSTER 2026-09-09 at generation 0, same body,
        # same reward, same seed: population mean 19.1 under the seed-policy
        # init against 110.1 under this one, where runs_repro2's own gen-0 Mean
        # was 107.69. That tree's `ga` verified at 1924.87; the seed-policy init
        # was flat at ~60-100 and falling at generation 250 of 320.
        #
        # THE COST IS REAL AND IS ACCEPTED: the gaussian pair no longer starts
        # from the same point as es/nes/RL, so "every arm begins at one point"
        # is false for this body. The alternative is a GA that cannot stand up,
        # which is not a comparison either.
        #
        # NOT SET FOR THE CHEETAH: the measurement above is about an ant falling
        # over, the cheetah's settings are provisional pending
        # probe_cheetah_stationary.sh, and a body-specific finding should not be
        # generalised by default.
        ne_arms=_ne_arms(ga_sigma=0.01, es_sigma=0.02, es_lr=0.005,
                         nes_sigma=0.02, nes_lr=0.0025,
                         ga_init_around_mean=False),
    ),
    'cheetah': dict(
        env_name='CheetahRun',
        # PROVISIONAL -- dm_control's constant for a body this is not. See the
        # module docstring and probe_cheetah_stationary.sh.
        target_speed=10.0,
        # brax's halfcheetah has a median per-dim observation std of 0.803
        # against the ant's 0.804 (measured 2026-09-09 over 200 random-action
        # steps), so 2.0 is ~2.5x the spread on both bodies and needs no
        # per-body correction. It is also what runs_repro2/mujoco used, which
        # is the cell where NE beat RL and the reason this value is kept.
        noise_range=2.0,
        # runs_repro2/mujoco/continual's own configs: GA sigma 0.1 /
        # elite_ratio 0.1, OpenES sigma 0.04 / lr 0.01, DNS iso 0.005 / line
        # 0.05 / k 3 with the AURORA settings above. Carried across the change
        # of simulator because sigma is a property of the BODY and the policy
        # parameterisation -- a (128, 128) tanh MLP over 17 observations and 6
        # actuators -- and neither changed. The ant's 0.01 would be a 10x
        # handicap here.
        #
        # NES HAS NO REPRO2 ROW: that tree has ga, es, dns, ppo, trac, redo and
        # cchain and no NES arm at all. It takes OpenES's sigma at half its
        # step, which is the relation the ant's swept row turned out to have
        # (0.02 / 0.0025 against OpenES's 0.02 / 0.005). PROVISIONAL, and the
        # first thing to sweep if the NES curve looks unlike the OpenES one.
        ne_arms=_ne_arms(ga_sigma=0.1, es_sigma=0.04, es_lr=0.01,
                         nes_sigma=0.04, nes_lr=0.005),
    ),
}

# The Slippery-Ant draw, `ANT_FRICTION_LOW_MULT=0.05` in the repro2 queue.
# One range for both bodies: it is a multiplier on the ground, not on the
# robot, so it means the same thing on either.
FRICTION_RANGE = (0.05, 5.0)

# The floor is per body; see the module docstring of this section. The ant
# takes the Slippery-Ant 0.05, the cheetah its own configuration's 0.2, below
# which its contact solver diverges rather than producing a hard task.
_FRICTION_LOW_BY_BODY = {'ant': FRICTION_RANGE[0], 'cheetah': 0.2}


def _friction_options(body):
    return {'friction_order': 'random',
            'friction_low': _FRICTION_LOW_BY_BODY.get(body, FRICTION_RANGE[0]),
            'friction_high': FRICTION_RANGE[1]}


_FRICTION_OPTIONS = {'friction_order': 'random',
                     'friction_low': FRICTION_RANGE[0],
                     'friction_high': FRICTION_RANGE[1]}

# cell -> (body, schedule, task_mod, extra task options)
#
# `task_mod` is what a sub-task VECTOR means and is the whole of the difference
# between the two families: an observation offset, or a ground-friction
# multiplier. Everything else -- the body, the reward, the policy, the budget,
# the phase grid -- is identical across a body's three cells, so the families
# differ in the perturbation and in nothing else.
CELLS = {}
for _body in BODIES:
    CELLS[f'{_body}_noise'] = (_body, 'switch', 'obs_noise', {})
    CELLS[f'{_body}_friction'] = (_body, 'switch', 'friction',
                                  dict(_friction_options(_body)))
    # A sub-task reverses the action map; the body, the world and the reward
    # are untouched. `source/envs/mjx.TaskSpec.act`.
    CELLS[f'{_body}_action'] = (_body, 'switch', 'action', {})
    # A sub-task is a TARGET SPEED, so the conflict is GRADED: a specialist at
    # either target is far outside the other's Gaussian window, while an
    # intermediate gait scores moderately on both and wins the worst case.
    # That is the structure friction lacks -- the ant re-routes around a
    # friction change (retention -79), so there is no trade-off for a
    # generalist to win. 2.0 vs 8.0 rather than the module default 0.5/2.0:
    # the window is half the target, so 0.5 and 2.0 overlap heavily and one
    # gait could satisfy both, measuring no conflict at all.
    CELLS[f'{_body}_speed'] = (_body, 'switch', 'speed',
                               {'speed_targets': '2.0,8.0'})
    # The reward ablation: `target_speed=None` is brax's stock unbounded v_x,
    # so the search maximises speed instead of holding one. Same sub-tasks as
    # the two continual families above, and a stationary cell to read them
    # against -- a speed-maximising run has a different ceiling AND a
    # different floor from a speed-tracking one, so its numbers cannot be put
    # beside the target-speed cells without it.
    CELLS[f'{_body}_noise_speedmax'] = (_body, 'switch', 'obs_noise',
                                        {'target_speed': None})
    CELLS[f'{_body}_friction_speedmax'] = (
        _body, 'switch', 'friction', dict(_friction_options(_body),
                                          target_speed=None))
    CELLS[f'{_body}_speedmax'] = (_body, 'task0', 'friction',
                                  dict(_friction_options(_body), target_speed=None))
    # The stationary control for BOTH families; see the docstring.
    CELLS[_body] = (_body, 'task0', 'friction', dict(_friction_options(_body)))
del _body

# The target-speed family with OVERLAPPING windows (2026-09-13): 2 vs 4 m/s.
# The reward window is half the target, so 2.0 and 8.0 share no gait and a
# generalist does not exist by construction (runs_ant_speed: no arm became
# one). At 2 vs 4 a gait near 3 m/s sits inside both windows, so a generalist
# is attainable while the sub-tasks still conflict.
CELLS['ant_speed24'] = ('ant', 'switch', 'speed', {'speed_targets': '2.0,4.0'})

CONTINUAL_CELLS = tuple(c for c, v in CELLS.items() if v[1] != 'task0')
NONCONTINUAL_CELLS = tuple(c for c, v in CELLS.items() if v[1] == 'task0')

NUM_PHASES = 20
# Ten sub-tasks over twenty phases: each visited exactly twice, the second
# visit being the retention measurement. The gymnax grid's structure.
NUM_TASKS = 10

# NE half of the matched budget.
NE_GENERATIONS = 320
NE_TASK_INTERVAL = NE_GENERATIONS // NUM_PHASES     # 16
NE_POP_SIZE = 512
NE_NUM_EVALS = 3                                    # pinned across arms
EPISODE_LENGTH = 1000

# RL half. `train_ppo.PPO_CONFIGS` owns these; repeated only so this file can
# assert the two agree (see `check()`).
RL_UPDATES = 48000
RL_TASK_INTERVAL = RL_UPDATES // NUM_PHASES         # 2400

# THE REPORTED CURVE, which is not the number either family selects on. The
# centroid and the population mean are re-scored on FRESH keys over this many
# episodes, every generation, on every sub-task.
#
# 10, matching the gymnax tree's `--report_episodes` and the MiniGrid study, so
# an mjx curve and a gymnax curve are the same estimator at the same episode
# count. The 2026-09-06 ant runs used 8 and the figures here do not mix with
# those.
#
# It is not part of the matched budget: 2 genomes x 10 sub-tasks x 10 episodes
# x 1000 steps is 2.0e5 a generation against the 1.536e6 the population costs,
# so 13%. That is the largest this overhead gets anywhere in the project (ten
# sub-tasks, thousand-step episodes) and it is identical for every NE arm, so
# it moves no comparison.
EVAL_EPISODES = 10

# ppo and the three continual-RL baselines, all on `run_ppo`'s one loop. No
# per-arm settings: they differ in the mechanism the runner switches on
# (TRAC's rescaling, ReDo's recycling, C-CHAIN's regulariser), never in the
# budget or the optimizer, or the comparison would not be about the mechanism.
# `pbt` (N = 8) and `pbt2` (N = 2): a population of the plain PPO learner over
# PPO's own budget, at two sizes
# (source/studies/generalists/train_ppo.py, the PBT block). Same runner,
# same budget, same schedule; it differs from `ppo` in the population.
# `pbt_weights` / `pbt2_weights`: the same populations WITHOUT explore
# (--pbt_mode weights_only, 2026-09-19); the paper's PBT rows are `full`.
RL_ARMS = ('ppo', 'trac', 'redo', 'cchain', 'pbt', 'pbt2',
           'pbt_weights', 'pbt2_weights')

NE_ARM_NAMES = ('nes', 'es', 'ga', 'dns_gaussian')
ARMS = NE_ARM_NAMES + RL_ARMS
# The order the figures list them in: NE first, then RL.
REPORTED_ARMS = ('ga', 'es', 'nes', 'dns_gaussian',
                 'ppo', 'trac', 'redo', 'cchain', 'pbt', 'pbt2')


def body_of(cell):
    return CELLS[cell][0]


def env_name(cell):
    """The suite environment a cell runs on."""
    return BODIES[body_of(cell)]['env_name']


def family(arm):
    if arm in NE_ARM_NAMES:
        return 'ne'
    if arm in RL_ARMS:
        return 'rl'
    raise ValueError(f'unknown arm {arm!r}; have {sorted(ARMS)}')


def ne_arm(cell, arm):
    """This body's settings for one NE arm, as a fresh dict."""
    spec = BODIES[body_of(cell)]['ne_arms'][arm]
    out = dict(spec)
    if 'searcher_kwargs' in out:
        out['searcher_kwargs'] = dict(out['searcher_kwargs'])
    return out


def task_options(cell):
    """The suite's `task_options` dict for one cell."""
    body, _schedule, task_mod, extra = CELLS[cell]
    return {'task_mod': task_mod,
            'target_speed': BODIES[body]['target_speed'], **extra}


def noise_range(cell):
    """Sigma of the observation offset; 0 where a sub-task is not one.

    Passed explicitly for the friction cells rather than left at the
    environment's default, so a finished friction run's config does not record
    an offset width it never drew.
    """
    body, _schedule, task_mod, _extra = CELLS[cell]
    return BODIES[body]['noise_range'] if task_mod == 'obs_noise' else 0.0


def check(cell):
    """Assert the two families really are matched. Called by the CLI on start.

    A silent mismatch here is the failure CLAUDE.md (c) is about: both arms
    finish, both write curves, and the figure compares a method against another
    method with more environment steps or a different number of task changes.
    Cheap to check, invisible if it is ever wrong.
    """
    from source.studies.generalists.train_ppo import PPO_CONFIGS

    hp = PPO_CONFIGS[env_name(cell)]
    ne_steps = NE_GENERATIONS * NE_POP_SIZE * NE_NUM_EVALS * EPISODE_LENGTH
    rl_steps = hp['num_updates'] * hp['num_envs'] * hp['num_steps']
    problems = []
    if ne_steps != rl_steps:
        problems.append(f'budget: NE {ne_steps:.3e} steps vs RL {rl_steps:.3e}')
    if hp['num_updates'] != RL_UPDATES:
        problems.append(f"updates: PPO_CONFIGS {hp['num_updates']} vs "
                        f'settings {RL_UPDATES}')
    if hp['num_updates'] // hp['task_interval'] != NUM_PHASES:
        problems.append(
            f"phases: RL {hp['num_updates'] // hp['task_interval']} vs "
            f'{NUM_PHASES}')
    if NE_GENERATIONS // NE_TASK_INTERVAL != NUM_PHASES:
        problems.append('phases: NE '
                        f'{NE_GENERATIONS // NE_TASK_INTERVAL} vs {NUM_PHASES}')
    if problems:
        raise SystemExit('mjx study settings are inconsistent:\n  '
                         + '\n  '.join(problems))
    return ne_steps
