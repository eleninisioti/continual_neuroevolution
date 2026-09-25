"""What the Kinetix study runs: the arms, their settings, and the budget.

This is a study, not a trainer. Neither of the two loops it drives lives here:

    NE   ``source/studies/generalists/train_nes.py:run_nes``
    RL   ``source/studies/generalists/train_ppo.py:run_ppo``

Both are suite-generic and touch Kinetix only through ``source/envs/kinetix.py``.
The eighteen files beside this one -- ``ga.py``, ``ga_continual.py``,
``dns_continual.py``, ``es_continual.py``, ``ppo.py`` and the rest, 500 KB of
them -- are the PREVIOUS codebase, kept only until the runs below reproduce
what they produced. Nothing here imports them, and nothing new should: a
second GA and a second PPO in the tree is exactly what CLAUDE.md (a) forbids.


## What this reproduces

`projects/kinetix/budget_g200_r3_gafinal` is the run that mattered: a GA that
solved all twenty hand-designed levels in the continual chain. Everything that
run fixed is fixed here at the same value --

    population        512
    generations/level 200
    rollouts/genome   3         (`evolve_reps`, which is what selected)  -- NOW 1, see below
    episode length    256       (frame skip 2)                           -- NOW 128, see below
    mutation sigma    0.5
    crossover rate    0.2
    archive           256 of 512 = elite_ratio 0.5
    policy            ActorOnlyPixelsRNN, recurrent off, 1,128,256 parameters

-- and the two sibling runs give the other NE arms theirs:
`budget_g200_r3_esfull_s1` (OpenES, sigma 0.02, lr 0.05) and
`budget_g200_r3_dnsfix_s1` (DNS, iso 0.005 / line 0.05, k 3).

Read those numbers off the runs' own configs, not retyped from a paper.


## The budget, corrected 2026-09-13: one rollout, 128 steps

Two things the working runs paid for and did not use, both MEASURED on the
finished stationary tree rather than argued:

* **The three rollouts per genome are one rollout three times.** A level
  resets to its stored state (`LevelSet.reset` hands Kinetix
  `override_reset_state`, and the key is never consumed), the physics is
  deterministic and the policy acts by argmax, so the same genome under six
  different keys scores identically to the last digit (`det_test`, the h0
  incumbent: -0.9714750051498413 six times; 3-eval mean == 1-eval value).
  `num_evals=3` bought no variance reduction; it re-ran the same episode.
* **Nothing happens after step 128.** Every solving GA incumbent re-scored
  at 96 / 128 / 160 / 192 / 256 steps returns the same number (8 of 20
  levels solved at every length), and the GA trained at `episode_length
  128` solves the same 20 of 20 stationary levels as at 256.

So the NE budget the RL arms were matched to, 200 x 512 x 3 x 256 =
7.864e7 steps a level, was six times what the search actually consumed,
and matching RL to the nominal number gave PPO six times the experience.
The grid below is the effective budget for BOTH families, by construction
rather than by an accounting factor: NE runs one rollout of 128 steps and
RL gets one sixth of its updates. The old tree (256 steps, 3 rollouts, RL
at 9600 updates a level) is `projects/iclr_2027/runs_kinetix`; the reported
one is `runs_kinetix_ep128_ev1`.


## The non-continual block, which is what runs first

Twenty stationary cells, one per level, each the SAME 200 generations the
continual chain gives that level. It answers the question that has to be
answered before any continual claim: **can each method solve a level at all,
on its own, at this budget?** A retention number for a method that never
learned the level in the first place says nothing.

It is also the FT reference: forward transfer subtracts each method's own
stationary curve from its continual one, so every arm in the continual block
needs its stationary twin at the same budget.

Levels nobody solves are not a failure of this block -- they are its output.
The continual chain warm-starts each level from the previous one, so a level
that is out of reach cold may well be in reach warm, and knowing WHICH levels
those are is the interesting half of the comparison.


## Compute-matched (CLAUDE.md (c))

Per level, in environment steps:

    NE   200 generations x 512 population x 1 rollout x 128 steps  = 1.311e7
    RL   1600 updates x 128 envs x 64 steps                        = 1.311e7

``check()`` recomputes both at the start of every trial and refuses to run if
they have drifted apart. The old codebase's RL runs were at 3.2e6 steps a
level against NE's 7.9e7 -- a 24x gap that made every cross-family number in
that table unquotable, and it is the one thing here that deliberately does
NOT reproduce them.

The RL shape (128 x 64 rather than the old runs' 16 x 1000) is throughput,
not a hyperparameter: rendering 125x125x3 frames is the cost of this body, and
16 parallel environments left the card idle. Everything PPO learns FROM --
lr 5e-5, gamma 0.995, gae_lambda 0.9, 8 epochs, 32 minibatches, clip 0.2,
ent_coef 0.01, vf_coef 0.5, grad norm 1.0 -- is the vendored
`third_party/kinetix/kinetix_config_pixels.yaml`, i.e. what Kinetix itself
tuned on this body.


## The RL arms work, and the bug that said otherwise

They are in the default arm set. Measured on h0_unicycle, from scratch: PPO
reaches the reported return 1.1 -- above the solved threshold -- by update 100,
which is 819,200 environment steps, against the 1.6e6 the previous codebase's
four from-scratch runs took to reach solve rate ~1.0. Entropy sits at 4.9-5.0
against the multi-discrete maximum of 5.78, i.e. healthy.

Recorded because the first version of this file said the opposite at length.
Before the fix, PPO reported `H=0.000` with the return at the floor and it was
written up here as an entropy collapse caused by Adam's step size across a
fan-in of 8,193. That was wrong. The cause was the multi-discrete action head:
it padded the ragged [3,3,3,3,2,2] logit block with -inf, which gives every
VALUE correctly and a NaN GRADIENT, so PPO's entropy bonus NaN'd the actor on
the first step -- and the head's own `isfinite` guard then read those NaN
logits back as an entropy of exactly zero.

The tell was that `H=0.023` survived a bisect over batch shape, a 10x learning
rate and a 1000x Adam epsilon, identical to three decimals. No training effect
is invariant to the learning rate. `actors._multi_discrete_grads_finite` is the
regression test, and both Kinetix queue scripts run it before queueing.

Two things that hypothesis produced and that the evidence then removed: an
`adam_eps` knob in the shared PPO (1e-5 and 1e-8 both solve the level, so
there was no case for it), and the plan to give this body a convolutional
critic. PPO's critic here is `ValueNetwork`, an MLP over the flat 46,876-value
observation, where Kinetix's own PPO shares a conv trunk between actor and
critic -- and it solves the level in half the old trainer's steps as it stands.
Revisit that only with a measurement that asks for it.


## No arm is told where a boundary is (CLAUDE.md (d))

Nothing here passes a boundary signal, and on the non-continual block there is
nothing to tell -- the level never changes. `--oracle` builds the deliberately
informed control and is the only way to turn any of it on.
"""

from __future__ import annotations

from source.envs.kinetix_levels import CELL_ALL, CELLS, LEVELS, cell_for

__all__ = ['LEVELS', 'CELLS', 'CELL_ALL', 'cell_for', 'NE_ARMS', 'RL_ARMS',
           'ARMS', 'family', 'check', 'CONTINUAL_CELLS', 'NONCONTINUAL_CELLS']


# ---------------------------------------------------------------------------
# The task grid
# ---------------------------------------------------------------------------

CONTINUAL_CELLS = (CELL_ALL,)
NONCONTINUAL_CELLS = tuple(cell_for(level) for level in LEVELS)

EPISODE_LENGTH = 128                 # 256 in the working runs; inert past 128 (see above)
NE_POP_SIZE = 512
NE_NUM_EVALS = 1                     # 3 in the working runs; the level is deterministic (see above)
GENERATIONS_PER_LEVEL = 200

# The continual chain: twenty phases of 200 generations, one per level.
NUM_TASKS = len(LEVELS)
NE_GENERATIONS_CONTINUAL = GENERATIONS_PER_LEVEL * NUM_TASKS        # 4000
NE_TASK_INTERVAL_CONTINUAL = GENERATIONS_PER_LEVEL                  # 200

# A stationary cell: the same 200 generations that level gets in the chain,
# cut into ten checkpoint phases. Nothing changes at those points -- the level
# is fixed -- and that is the point: a dormancy, rank or churn number only says
# something about task CHANGES if the same measurement on a run of the same
# length with no change does something different.
NE_GENERATIONS = GENERATIONS_PER_LEVEL                              # 200
NONCONTINUAL_PHASES = 10
NE_TASK_INTERVAL = NE_GENERATIONS // NONCONTINUAL_PHASES            # 20

# RL half of the matched budget. `train_ppo.PPO_CONFIGS` owns the shape;
# repeated here only so `check()` can assert the two halves agree.
RL_NUM_ENVS = 128
RL_NUM_STEPS = 64
RL_UPDATES = 1600                    # 9600 against the nominal budget; one sixth, see above
RL_UPDATES_CONTINUAL = RL_UPDATES * NUM_TASKS
RL_TASK_INTERVAL = RL_UPDATES // NONCONTINUAL_PHASES                # 160
RL_TASK_INTERVAL_CONTINUAL = RL_UPDATES                             # 1600

# THE REPORTED CURVE, which is not the number the search selects on: the
# centroid and the population mean re-scored on FRESH keys with argmax actions.
# 10, as the gymnax and MiniGrid trees use, so one figure script reads all
# three bodies. Not part of the matched budget -- 2 genomes x 1 level x 10
# episodes x 128 steps is 2,560 steps a generation against the 65,536 the
# population costs, 3.9%. On this body the ten are ten copies of one
# deterministic rollout (see above); kept at 10 so one figure script reads
# all three bodies.
EVAL_EPISODES = 10


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------
#
# `family` picks the runner; everything else is passed straight through.
# `searcher_kwargs` goes to `ne.build_searcher`.

NE_ARMS = {
    # THE ARM THIS BLOCK EXISTS FOR. Every value is
    # `budget_g200_r3_gafinal`'s: sigma 0.5, crossover 0.2, and an archive of
    # 256 out of 512 (`num_elites` in that run's log = elite_ratio 0.5).
    #
    # `sigma` is the top-level name, NOT `sigma_init` inside searcher_kwargs:
    # `run_nes` forwards its own `sigma` as `sigma_init`, so passing both is a
    # duplicate keyword and the run dies at construction.
    'ga': dict(method='ga', sigma=0.5,
               searcher_kwargs=dict(elite_ratio=0.5, cross_over_rate=0.2)),

    # OpenES at `budget_g200_r3_esfull_s1`'s settings -- sigma 0.02, lr 0.05 --
    # with the two knobs that DEFINE OpenES here (centered ranks, Adam), which
    # is how `es` is spelled on every other body in this tree.
    'es': dict(method='openes', sigma=0.02, learning_rate=0.05,
               optimizer='adam', shaping='centered_rank'),
    #   es_zscore       `es` with z-scored fitness instead of centred ranks,
    #                   Adam kept. Why (2026-09-21): under ranks one solver
    #                   among 512 samples gets u = +0.5 and barely moves the
    #                   mean, the stated reason for es_hold; a z-score gives it
    #                   u ~ sqrt(P). Does shaping alone close the gap es_hold
    #                   closes (16.4 -> 19.2 levels solved at the switch)?
    #                   `nes` (z-score + SGD) solved 8.6, but that also changes
    #                   the optimizer.
    'es_zscore': dict(method='openes', sigma=0.02, learning_rate=0.05,
                      optimizer='adam', shaping='zscore'),
    #   es_lr0.02 / es_lr0.01   `es` with a smaller step. Why (2026-09-21,
    #                   es_zscore lost to es_hold on levels 0-4): Adam moves
    #                   every coordinate by ~lr, so the mean steps
    #                   ~lr*sqrt(1.13M) = 53 at lr 0.05 while the samples it
    #                   was scored on sit sigma*sqrt(1.13M) = 21 away -- the
    #                   mean lands where no sample was tested, one reading of
    #                   "samples solve, the mean does not". Only the setting
    #                   changes; if neither matches es_hold the paper keeps it.
    'es_lr0.02': dict(method='openes', sigma=0.02, learning_rate=0.02,
                      optimizer='adam', shaping='centered_rank'),
    'es_lr0.01': dict(method='openes', sigma=0.02, learning_rate=0.01,
                      optimizer='adam', shaping='centered_rank'),

    # OpenES whose MEAN is meant to hold a level until the switch
    # (`openes_adaptive`, source/studies/generalists/ne.py): the best sample
    # seen is kept and the mean jumps to it when it falls far behind; the
    # sampling width and step shrink while the mean beats most samples; both
    # reopen when the mean's own score collapses. Why (the ep128_ev1 chain,
    # 2026-09-14): OpenES's mean solved 16.4/20 levels at the switch while its
    # best final-generation sample solved 18.6 -- samples found solutions the
    # mean never reached or did not keep. Two of the 512 evaluations score the
    # mean and the kept sample.
    #   es_jump         the keep-and-jump part alone (no settle, no reopen):
    #                   on the toys it never hurt and sped the mean's tracking
    #                   (min-wells k8, gen 50: 0.99 vs 0.56); the gated settle
    #                   cost a little on the 8-coordinate ripple (found 0.83).
    'es_jump': dict(method='openes_adaptive', sigma=0.02, learning_rate=0.05,
                    optimizer='adam', shaping='centered_rank',
                    searcher_kwargs=dict(jump_kappa=3.0)),
    #   es_hold         keep-and-jump plus the verified step: the mean moves to
    #                   a proposed step only once it has scored no worse than
    #                   the mean (within one sample spread). Why: es_jump's mean
    #                   alternated 1010... on and off the kept solver on 10/12
    #                   GH200 runs (one ES step off it, jump back, repeat).
    'es_hold': dict(method='openes_adaptive', sigma=0.02, learning_rate=0.05,
                    optimizer='adam', shaping='centered_rank',
                    searcher_kwargs=dict(jump_kappa=3.0, accept_tol=1.0)),
    'es_adaptive': dict(method='openes_adaptive', sigma=0.02, learning_rate=0.05,
                        optimizer='adam', shaping='centered_rank',
                        searcher_kwargs=dict(jump_kappa=3.0, settle_rate=0.1,
                                             restart_drop=0.1, restart_hold=50)),

    # Plain NES: the same searcher, standardized fitness and SGD. No Kinetix
    # run of it exists, so it takes OpenES's width and the SGD learning rate
    # the other bodies pair with `zscore`. Treat its first numbers as a pilot.
    'nes': dict(method='nes', sigma=0.02, learning_rate=0.05,
                optimizer='sgd', shaping='zscore'),

    # DNS at `budget_g200_r3_dnsfix_s1`'s settings: Iso+LineDD variation,
    # iso 0.005 / line 0.05, k 3.
    #
    # The descriptor is the DUTY FACTOR -- what fraction of the episode each of
    # the six actuator bindings was driven (`kinetix.util.behaviour`). Not
    # AURORA, which is the default on the other bodies: AURORA's input here is
    # a 13-value per-step feature vector rather than the observation, so it is
    # available (`--dns_descriptor aurora`) but it is a second thing to trust
    # in a block whose job is to establish that the arms work at all. Duty
    # factor is the descriptor that means the same thing on all twenty levels.
    'dns': dict(method='dns',
                searcher_kwargs=dict(iso_sigma=0.005, line_sigma=0.05, k=3,
                                     descriptor='handcrafted')),

    # The GAUSSIAN pair: `ga` and this differ in the SELECTION RULE alone,
    # which is the arm the ICLR grid reports on the other bodies. Identical
    # mutation to `ga` above -- sigma 0.5, crossover 0.2 -- so the comparison
    # is not confounded by the operator.
    'dns_gaussian': dict(method='dns_gaussian', sigma=0.5,
                         searcher_kwargs=dict(cross_over_rate=0.2, k=3,
                                              descriptor='handcrafted')),

    # CENTROID-TRACKING PILOT (2026-09-13, uncommitted). The GA above with one
    # rule added that acts only while the archive's centroid scores below its
    # median elite (source/studies/generalists/ne.py). Same sigma, crossover
    # and archive as `ga`, so a difference is the rule alone.
    #   ga_merge_noise  rank noise on selection, ramped fast enough to act
    #                   inside 200 generations (the toy's default ramp needs
    #                   ~200-250 generations to merge a split archive)
    #   ga_track        sigma shrinks while the centroid lags, grows back
    #                   (never past 0.5) once it does not
    'ga_merge_noise': dict(method='ga_merge_noise', sigma=0.5,
                           searcher_kwargs=dict(elite_ratio=0.5,
                                                cross_over_rate=0.2,
                                                merge_rate=0.1, merge_max=2.0)),
    'ga_track': dict(method='ga_track', sigma=0.5,
                     searcher_kwargs=dict(elite_ratio=0.5, cross_over_rate=0.2,
                                          track_rate=0.3)),
    #   ga_merge_track  both at once, from the one centroid signal: rank noise
    #                   (fast ramp) and the sigma shrink
    'ga_merge_track': dict(method='ga_merge_track', sigma=0.5,
                           searcher_kwargs=dict(elite_ratio=0.5,
                                                cross_over_rate=0.2,
                                                merge_rate=0.1, merge_max=2.0,
                                                sigma_rate=0.1)),
    #   ga_merge_track_fast  the same with the faster sigma shrink (0.3): on
    #                   the min-wells toy the only setting that tracked by
    #                   generation 200 at both k = 2 and k = 8, at the cost of
    #                   an early dip in the elite at k = 8
    #   ga_focus        consolidation through the choice of parents: while the
    #                   centroid lags, breed only from the best few archive
    #                   members (plus the sigma shrink); the pilot showed rank
    #                   noise alone cannot merge Kinetix's many unrelated
    #                   solver lineages inside 200 generations on the hardest
    #                   levels (unicycle, thrustcontrol_right)
    'ga_focus': dict(method='ga_focus', sigma=0.5,
                     searcher_kwargs=dict(elite_ratio=0.5, cross_over_rate=0.2,
                                          focus_rate=0.3, sigma_rate=0.1)),
    #   ga_focus_purge  the focus also bars old archive members outside the
    #                   focused pool from surviving: ga_focus alone kept the
    #                   old unrelated solvers, whose slightly higher returns
    #                   outranked the focused children
    'ga_focus_purge': dict(method='ga_focus_purge', sigma=0.5,
                           searcher_kwargs=dict(elite_ratio=0.5,
                                                cross_over_rate=0.2,
                                                focus_rate=0.3,
                                                sigma_rate=0.1)),
    #   ga_focus_purge_t90  the same with a stricter tracking signal: the
    #                   centroid must score at least as well as 90% of the
    #                   archive, not the median. Under the purge the archive
    #                   fills with failing children, the centroid then equals
    #                   the median, and the 0.5 target stopped the sigma
    #                   shrink at ~0.2 (home pilot, 2026-09-14)
    'ga_focus_purge_t90': dict(method='ga_focus_purge', sigma=0.5,
                               searcher_kwargs=dict(elite_ratio=0.5,
                                                    cross_over_rate=0.2,
                                                    focus_rate=0.3,
                                                    sigma_rate=0.1,
                                                    track_target=0.9)),
    #   ga_focus_purge_t70  target 0.7: on the toys it tracks the elite as
    #                   t90 does but, unlike t90, still finds the rugged
    #                   landscape's generalist (t90 found it in 1 of 12 seeds)
    'ga_focus_purge_t70': dict(method='ga_focus_purge', sigma=0.5,
                               searcher_kwargs=dict(elite_ratio=0.5,
                                                    cross_over_rate=0.2,
                                                    focus_rate=0.3,
                                                    sigma_rate=0.1,
                                                    track_target=0.7)),
    #   ga_focus_fine   ga_focus (no purge) with the sigma floor at 1e-5 and
    #                   target 0.9. Why: the probe of ga_consolidate_diag's
    #                   final elites (2026-09-14) -- evaluation is deterministic,
    #                   children solve 88-97% at sigma 1e-4 but 0-12% at the old
    #                   floor 5e-3 (sigma_init / 100), and the purge at a
    #                   one-member pool left no selection (pool + 255 children
    #                   = the 256 slots, failing children kept). With solving
    #                   children, their returns tie the elite's and displace
    #                   the old unrelated solvers without any purge.
    #   ga_focus_fine_fast  sigma_rate 0.3: ~30 generations from 0.5 to 1e-4
    #                   rather than ~95
    #   ga_keep         the `ga` arm with one change: a child must strictly beat
    #                   an archive member to replace it (ties keep the parent).
    #                   Why: on the smooth toy the GA found the generalist and
    #                   lost it by drifting across tied scores; ga_keep kept it
    #                   in every seed, sigma and interval (2026-09-14).
    'ga_keep': dict(method='ga_keep', sigma=0.5,
                    searcher_kwargs=dict(elite_ratio=0.5, cross_over_rate=0.2)),
    'ga_focus_fine': dict(method='ga_focus', sigma=0.5,
                          searcher_kwargs=dict(elite_ratio=0.5,
                                               cross_over_rate=0.2,
                                               focus_rate=0.3,
                                               sigma_rate=0.1,
                                               track_target=0.9,
                                               sigma_min=1e-5)),
    #   ga_focus_explore  ga_focus_fine plus explorers: a quarter of the
    #                   offspring always bred from the whole archive at sigma
    #                   0.5. For the chain: without them a population
    #                   consolidated on one level cannot search the next.
    'ga_focus_explore': dict(method='ga_focus', sigma=0.5,
                             searcher_kwargs=dict(elite_ratio=0.5,
                                                  cross_over_rate=0.2,
                                                  focus_rate=0.3,
                                                  sigma_rate=0.1,
                                                  track_target=0.9,
                                                  sigma_min=1e-5,
                                                  explore_fraction=0.25)),
    #   ga_focus_explore_nox  ga_focus_explore without crossover, the plain
    #                   GA's operator: does the paper's variant need it?
    #                   (probe, 2026-09-21, Kinetix20 + MountainCar noise)
    'ga_focus_explore_nox': dict(method='ga_focus', sigma=0.5,
                                 searcher_kwargs=dict(elite_ratio=0.5,
                                                      cross_over_rate=0.0,
                                                      focus_rate=0.3,
                                                      sigma_rate=0.1,
                                                      track_target=0.9,
                                                      sigma_min=1e-5,
                                                      explore_fraction=0.25)),
    #   ga_focus_restart  ga_focus_fine that searches again when its archive's
    #                   own re-scored fitness collapses (over half the members
    #                   more than 10% below their stored score): sigma and
    #                   focus back to 0.5 and 1 for 50 generations. The other
    #                   answer to the chain's problem, without explorers.
    'ga_focus_restart': dict(method='ga_focus', sigma=0.5,
                             searcher_kwargs=dict(elite_ratio=0.5,
                                                  cross_over_rate=0.2,
                                                  focus_rate=0.3,
                                                  sigma_rate=0.1,
                                                  track_target=0.9,
                                                  sigma_min=1e-5,
                                                  restart_share=0.5,
                                                  restart_drop=0.1,
                                                  restart_hold=50)),
    'ga_focus_fine_fast': dict(method='ga_focus', sigma=0.5,
                               searcher_kwargs=dict(elite_ratio=0.5,
                                                    cross_over_rate=0.2,
                                                    focus_rate=0.3,
                                                    sigma_rate=0.3,
                                                    track_target=0.9,
                                                    sigma_min=1e-5)),
    #   ga_consolidate_diag  DIAGNOSTIC, not a method: focus and sigma driven
    #                   to their floors within a few generations (target 1.0),
    #                   to see whether a population consolidated on the elite
    #                   at small sigma makes the centroid solve at all
    'ga_consolidate_diag': dict(method='ga_focus_purge', sigma=0.5,
                                searcher_kwargs=dict(elite_ratio=0.5,
                                                     cross_over_rate=0.2,
                                                     focus_rate=5.0,
                                                     sigma_rate=2.0,
                                                     track_target=1.0)),
    'ga_focus_noise': dict(method='ga_focus', sigma=0.5,
                           searcher_kwargs=dict(elite_ratio=0.5,
                                                cross_over_rate=0.2,
                                                focus_rate=0.3, sigma_rate=0.1,
                                                merge_rate=0.1, merge_max=2.0)),
    'ga_merge_track_fast': dict(method='ga_merge_track', sigma=0.5,
                                searcher_kwargs=dict(elite_ratio=0.5,
                                                     cross_over_rate=0.2,
                                                     merge_rate=0.1,
                                                     merge_max=2.0,
                                                     sigma_rate=0.3)),
}

# PPO and the three continual-RL baselines, all on `run_ppo`'s one loop. They
# differ in the mechanism the runner switches on (TRAC's rescaling, ReDo's
# recycling, C-CHAIN's regulariser), never in the budget or the optimizer.
# `pbt` (N = 8) and `pbt2` (N = 2): a population of the plain PPO learner over
# PPO's own budget, at two sizes
# (source/studies/generalists/train_ppo.py, the PBT block). Same runner,
# same budget, same schedule; it differs from `ppo` in the population.
# `pbt_weights` / `pbt2_weights`: the same populations WITHOUT explore
# (--pbt_mode weights_only, 2026-09-19); the paper's PBT rows are `full`.
RL_ARMS = ('ppo', 'trac', 'redo', 'cchain', 'pbt', 'pbt2',
           'pbt_weights', 'pbt2_weights')

ARMS = dict(NE_ARMS)
ARMS.update({name: None for name in RL_ARMS})


def family(method):
    if method in NE_ARMS:
        return 'ne'
    if method in RL_ARMS:
        return 'rl'
    raise KeyError(f'unknown Kinetix arm {method!r}; have {sorted(ARMS)}')


def check(continual=False):
    """The compute match, recomputed. Returns the per-level step budget.

    Raises if the two families have drifted apart. Called at the start of
    every trial, because a comparison that is not matched is not a comparison
    and a comment saying it is matched is not a check.
    """
    ne_steps = (NE_GENERATIONS * NE_POP_SIZE * NE_NUM_EVALS * EPISODE_LENGTH)
    rl_steps = RL_UPDATES * RL_NUM_ENVS * RL_NUM_STEPS
    if ne_steps != rl_steps:
        raise AssertionError(
            f'Kinetix budgets have drifted: NE {ne_steps:.4e} environment '
            f'steps against RL {rl_steps:.4e}')
    phases_ne = NE_GENERATIONS // NE_TASK_INTERVAL
    phases_rl = RL_UPDATES // RL_TASK_INTERVAL
    if phases_ne != phases_rl:
        raise AssertionError(
            f'Kinetix phase grids have drifted: NE {phases_ne} phases against '
            f'RL {phases_rl}; both families must meet a checkpoint at the '
            'same point on the shared wall of steps (CLAUDE.md (c))')
    if NE_POP_SIZE % 32:
        raise AssertionError(
            f'population {NE_POP_SIZE} is not divisible by the Kinetix '
            'eval_batch_size (32); the chunked scoring pass needs it to be')
    # The suite table is what `build_env` reads; this module is what the
    # queue scripts and the arithmetic above read. One number, checked here
    # rather than kept equal by hand (lazy import: the suite pulls in jax).
    from source.envs.kinetix import ENV_CONFIGS
    table = int(ENV_CONFIGS[CELL_ALL]['episode_length'])
    if table != EPISODE_LENGTH:
        raise AssertionError(
            f'source/envs/kinetix.py has episode_length {table} against '
            f'settings.EPISODE_LENGTH {EPISODE_LENGTH}; the budget is computed '
            'from the wrong number')
    if continual:
        return ne_steps * NUM_TASKS
    return ne_steps
