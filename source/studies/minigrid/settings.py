"""What the MiniGrid study runs: the arms, their settings, and the budget.

This is a study, not a trainer. Neither of the two loops it drives lives here:

    NE   ``source/studies/generalists/train_nes.py:run_nes`` -- one ask/tell
         loop over the four searchers in ``source/studies/generalists/ne.py``.
    RL   ``source/studies/generalists/train_ppo.py:run_ppo`` -- PPO and the
         three continual-RL baselines built on top of it.

Both are already suite-generic: they resolve an environment through
``source/envs/registry.py`` and touch MiniGrid only through
``source/envs/minigrid.py``. Forking them into a `train_*_minigrid.py` per
method -- the shape ``source/studies/gymnax/`` has -- would put a second
implementation of NES and of PPO in the tree, which is exactly what CLAUDE.md
(a) forbids. So a body is a settings table and an output layout, and that is
all this file is.

(The package they currently sit under is a misnomer: they are the shared
runners, not the generalists study's private code. Moving them somewhere
neutral is the "collapse the 33 trainers" step and is deliberately NOT done
here -- it would touch every reported generalists path while gymnax runs are
in flight.)


## The arms

The eight the ICLR grid settled on, named exactly as the gymnax tree names
them so one figure script reads both bodies:

    ga  es  nes  dns_gaussian  ppo  trac  redo  cchain

`es` is OpenES; `ga` and `dns_gaussian` are the GAUSSIAN pair, differing in
the selection rule alone. The Iso+LineDD arms (`ga_isoline`, `dns`) are the
operator ablation and are NOT in the paper, so they are not here either.


## Where the numbers come from, and where they do not

Every NE setting below is the value the 2026-09-07/08 MiniGrid sweeps settled
on for THIS body (`projects/generalists/runs/minigrid_*_sweep_*`), read back
off those runs' configs rather than retyped from a paper. Sigma does not
transfer between bodies -- the GA mutates at 0.01 here against 0.5 on gymnax,
and copying the gymnax row across would be a handicap, not a control.

The ONE place this departs from those runs is `num_evals`, and it is a
correction rather than a preference. The sweeps gave the GA 12 rollouts per
individual against NES's 3, which is 4x the environment steps for one arm of a
comparison that claims to be compute-matched (CLAUDE.md (c)). Every NE arm
here is pinned to the same number, as `block_gymnax_continual` pins it for the
same reason.


## The budget, and why the two families' numbers look nothing alike

Compute-matched, in environment steps:

    NE   4000 generations x 512 population x 3 rollouts x 1024 steps = 6.29e9
    RL   61,440 updates x 2048 envs x 50 steps                       = 6.29e9

and split into the SAME twenty phases -- 200 generations against 3072 updates
-- so both families meet a task change at the same point on that shared wall
of steps, which is CLAUDE.md (c)'s second half. The RL half of that arithmetic
is `train_ppo.PPO_CONFIGS['MiniGrid']`; it is repeated here only as the phase
count, so the two cannot drift.

1024 is the SCAN length, not a cap: each environment keeps its own max_steps
(256 on the 8x8 room, 1024 on the 16x16 one) because MiniGrid's reward is
`1 - 0.9 t / max_steps` and the cap is part of it.


## The cells

    MiniGrid_8x8_16x16   continual. Two sub-tasks, alternating every phase:
                         8x8, 16x16, 8x8, ... Twenty phases means each is seen
                         ten times, so a revisit measures retention -- the
                         same structure as the gymnax grid's NUM_TASKS=20
                         TASK_PERIOD=10.
    MiniGrid_8x8         stationary control, the 8x8 room throughout.
    MiniGrid_16x16       stationary control, the 16x16 room throughout.

The two rooms are nested -- the 16x16 specialist solves both (0.95 / 0.97) and
the 8x8 specialist reaches 0.73 on the 16x16 -- so a generalist exists and the
question "does switching cost you one?" is answerable. `source/envs/minigrid.py`
records the probes that chose this pair over the key-and-door tasks, where
every NE arm scores a flat zero and no comparison is possible.

The stationary cells are checkpointed on the SAME 200-generation grid as the
continual one even though nothing changes at those points. It costs nothing
and it is what lets the plasticity figure have a no-task-change control: a
dormancy or rank number is only evidence about task changes if the same
measurement on a stationary run of the same length does something different.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The task grid
# ---------------------------------------------------------------------------

ENV_NAME = 'MiniGrid'                 # the suite entry in source/envs/minigrid.py
ROOMS = ('EmptyRandom-8x8', 'EmptyRandom-16x16')

# cell -> (schedule, the sub-task pair the cell is built from)
#
# A stationary cell still names BOTH rooms and then pins the schedule to one of
# them, rather than naming one room twice. `build_env` refuses a single-element
# pair, and -- more importantly -- it keeps the observation encoding, the
# episode scan length and the policy identical across all three cells, so a
# stationary run and a switching run are the same experiment minus the switch.
CELLS = {
    'MiniGrid_8x8_16x16': ('switch', ROOMS),
    'MiniGrid_8x8':       ('task0',  ROOMS),
    'MiniGrid_16x16':     ('task1',  ROOMS),
}

CONTINUAL_CELLS = ('MiniGrid_8x8_16x16',)
NONCONTINUAL_CELLS = ('MiniGrid_8x8', 'MiniGrid_16x16')

NUM_PHASES = 20
NUM_TASKS = 2

# NE half of the matched budget.
NE_GENERATIONS = 4000
NE_TASK_INTERVAL = NE_GENERATIONS // NUM_PHASES     # 200
NE_POP_SIZE = 512
NE_NUM_EVALS = 3                                    # pinned; see the docstring
EPISODE_LENGTH = 1024

# RL half. `train_ppo.PPO_CONFIGS['MiniGrid']` owns these; repeated only so
# this file can assert the phase counts agree (see `check()` below).
RL_UPDATES = 61440
RL_TASK_INTERVAL = RL_UPDATES // NUM_PHASES         # 3072

# THE REPORTED CURVE, which is not the number either family selects on. The
# centroid and the population mean are re-scored on FRESH keys over this many
# episodes with argmax actions, every generation, on every sub-task.
#
# 10, matching the gymnax tree's `--report_episodes`, so a MiniGrid curve and a
# gymnax curve are the same estimator at the same episode count. The MiniGrid
# runs of 2026-09-07/08 used 16 (the shared runner's default) and the figures
# here do not mix with those.
#
# NOT `num_evals`, and the distinction is the point: the search's own column is
# a max over the population of a `num_evals` mean whose FIRST draw did the
# selecting, so it carries the winner's curse -- by a different amount per
# method, which does not cancel in a comparison. This evaluation feeds nothing
# back into the search.
#
# It is also not part of the matched budget: 2 genomes x 2 sub-tasks x 10
# episodes is 20,480 steps a generation against the 1,572,864 the population
# costs, so 1.3% -- the same standing the gymnax trainers give it.
EVAL_EPISODES = 10


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------
#
# `family` picks the runner; everything else is passed straight through to it.
# `searcher_kwargs` goes to `ne.build_searcher`, which is the one place that
# knows which knobs define which method.

NE_ARMS = {
    # NES: standardized fitness, plain SGD. The sweep's cell.
    'nes': dict(method='nes', sigma=0.1, learning_rate=0.05,
                optimizer='sgd', shaping='zscore'),
    # OpenES: the SAME searcher with the two settings that define it --
    # centered ranks and Adam. Named `es` to match the gymnax tree.
    'es': dict(method='openes', sigma=0.1, learning_rate=0.02,
               optimizer='adam', shaping='centered_rank'),
    # The gaussian pair. Identical mutation (sigma_init 0.01, no crossover),
    # so `ga` vs `dns_gaussian` isolates the SELECTION RULE and nothing else.
    # `sigma` is the top-level name, NOT `sigma_init` in searcher_kwargs:
    # `run_nes` already forwards its own `sigma` as `sigma_init`, so passing
    # both is a duplicate keyword and the run dies at construction. It is also
    # the name a finished run's config records the mutation width under, which
    # is how the sweep's value is read back.
    'ga': dict(method='ga', sigma=0.01,
               searcher_kwargs=dict(elite_ratio=0.1)),
    'dns_gaussian': dict(
        method='dns_gaussian',
        # `descriptor` and the aurora knobs are searcher_kwargs too --
        # `run_nes` pops them out of that dict rather than taking them as
        # arguments, which is also how they appear in a finished run's config.
        #
        # AURORA descriptors, as the gymnax dns_gaussian runs use, at this
        # body's trajectory length: a MiniGrid episode is 1024 steps against
        # gymnax's 500 and the encoder sees a 1225-value view, so 10 steps
        # rather than 50. Both from the 2026-09-07 minigrid_dnsprobe runs.
        sigma=0.01,
        searcher_kwargs=dict(cross_over_rate=0.0, k=3,
                             descriptor='aurora', traj_steps=10,
                             aurora_latent_dim=6, aurora_train_ratio=8)),
}

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

ARMS = tuple(NE_ARMS) + RL_ARMS
# The order the figures list them in: NE first, then RL.
REPORTED_ARMS = ('ga', 'es', 'nes', 'dns_gaussian',
                 'ppo', 'trac', 'redo', 'cchain', 'pbt', 'pbt2')


def family(arm):
    if arm in NE_ARMS:
        return 'ne'
    if arm in RL_ARMS:
        return 'rl'
    raise ValueError(f'unknown arm {arm!r}; have {sorted(ARMS)}')


def check():
    """Assert the two families really are matched. Called by the CLI on start.

    A silent mismatch here is the failure CLAUDE.md (c) is about: both arms
    finish, both write curves, and the figure compares a method against
    another method with more environment steps or a different number of task
    changes. Cheap to check, invisible if it is ever wrong.
    """
    from source.studies.generalists.train_ppo import PPO_CONFIGS

    hp = PPO_CONFIGS[ENV_NAME]
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
        raise SystemExit('MiniGrid study settings are inconsistent:\n  '
                         + '\n  '.join(problems))
    return ne_steps
