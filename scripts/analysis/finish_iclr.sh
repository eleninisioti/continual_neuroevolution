#!/bin/bash
# ============================================================================
# Everything that turns ONE run tree into the paper's figures and tables,
# under projects/iclr_2027/paper/<suite>/<family>/ -- on gymnax, one directory
# per perturbation with the schedules under it: paper/gymnax/noise/{10task,2task},
# paper/gymnax/physics/{10task,2task}, paper/gymnax/actions/2task; on mjx the
# body comes first: paper/mjx/<body>/<perturbation>/<schedule>.
#
#     scripts/analysis/finish_iclr.sh noise      # gymnax, observation offset
#     scripts/analysis/finish_iclr.sh freq50     # gymnax noise, switch every 50 gens
#     scripts/analysis/finish_iclr.sh freq400    # gymnax noise, switch every 400 gens
#     scripts/analysis/finish_iclr.sh physics    # gymnax, body rescaling
#     scripts/analysis/finish_iclr.sh actions    # gymnax, action reversal
#     scripts/analysis/finish_iclr.sh noise_2task    # gymnax, observation offset, TWO sub-tasks
#     scripts/analysis/finish_iclr.sh physics_2task  # gymnax, body rescaling, TWO sub-tasks
#     scripts/analysis/finish_iclr.sh minigrid   # MiniGrid, 8x8 / 16x16 rooms
#     scripts/analysis/finish_iclr.sh kinetix    # Kinetix, 20 medium levels
#     scripts/analysis/finish_iclr.sh cheetah_noise     # MJX cheetah, observation offset
#     scripts/analysis/finish_iclr.sh cheetah_friction  # MJX cheetah, ground friction
#     scripts/analysis/finish_iclr.sh ant        # brax ant (mjx), ground friction
#     scripts/analysis/finish_iclr.sh ant_2task         # the ant, TWO sub-tasks
#     scripts/analysis/finish_iclr.sh ant_warmup        # the ant, sub-task 0 held for half the budget
#     scripts/analysis/finish_iclr.sh ant_2task_warmup  # both
#     scripts/analysis/finish_iclr.sh ant_speed         # the ant, target speed 2 vs 8 m/s
#     scripts/analysis/finish_iclr.sh ant_speed_warmup  # the same, 2 m/s held for half the budget
#     scripts/analysis/finish_iclr.sh cheetah_noise05_t10  # cheetah offset 0.5, 10 sub-tasks (the headline)
#     scripts/analysis/finish_iclr.sh cheetah_noise025     # cheetah offset 0.25, 2 sub-tasks
#     scripts/analysis/finish_iclr.sh ant_noise025         # ant offset 0.25, 2 sub-tasks, warm-up
#     scripts/analysis/finish_iclr.sh cheetah_action       # cheetah action reversal, 2 sub-tasks
#     scripts/analysis/finish_iclr.sh ant_action           # ant action reversal, 2 sub-tasks
#     scripts/analysis/finish_iclr.sh ant_noise025_t10     # ant offset 0.25, 10 sub-tasks, warm-up
#     scripts/analysis/finish_iclr.sh ant_friction_2task   # ant friction, 2 sub-tasks, warm-up
#     scripts/analysis/finish_iclr.sh cheetah_friction_2task  # cheetah friction, 2 sub-tasks
#
# One script for every family and every body (CLAUDE.md rule (a)): they differ
# in the run tree, the reported cells and the output directory, and in
# NOTHING else -- same arms, same budget, same passes, same figure code. The
# three scripts this replaces (finish_iclr_centroid.sh, finish_iclr_param.sh,
# finish_iclr_actions.sh) had drifted: only the first took `--cells` and
# `--agent`, so the other two could not draw the centroid plasticity figure and
# could not draw a grid with a different sigma per environment.
#
# A MiniGrid tree is written by the SHARED runners, whose column names differ
# from the gymnax trainers'. Rather than teach every figure script a second
# vocabulary, `scripts/analysis/migrate_shared_runner_columns.py` renames the
# tree in place once (idempotent), and this script runs it under `verify`.
#
# The passes, in order:
#   verify    both phases are complete, compute-matched and boundary-aligned
#             (CLAUDE.md rule (c)). Fatal unless ALLOW_INCOMPLETE=1, in which
#             case the figures are drawn over what is there and the gaps are
#             named in verify's own output -- read it before reading them.
#   evaluate  the post-hoc pass that fills the ZT column: every sub-task's
#             saved agent, re-rolled on its own sub-task, on the NEXT one
#             before any search has touched it, and on the PREVIOUS one. GPU.
#   diverge   the pass that fills F and BD: forgetting, and the fraction of a
#             sub-task agent's visited states on which the next sub-task's
#             agent acts differently. GPU.
#   dormancy  per-unit dormant masks at one checkpoint per sub-task, which is
#             what tells a DEAD unit from a merely sparse one. CPU.
#   plastic   the plasticity figure and its persistence panel. CPU.
#   plot      the lineplot, legend and table, at both reported metrics, plus
#             the stationary reference and the metrics figure (the table's
#             columns as points with 95% bootstrap CIs).
#   generalist  is the saved agent a generalist (clears the threshold on the
#             sub-task it was just trained on AND on the previous one) or a
#             switching specialist? Found / Held / Retention and the phase
#             outcomes, from the `evaluate` pass. Needs a solved threshold:
#             MiniGrid uses 0.8 (see the family table); the gymnax families
#             take the registry's. CPU.
#
# STEPS selects passes: STEPS="verify dormancy plastic plot" is the cheap set
# that needs no GPU and draws every figure with F/BD/ZT left as `--`.
#
# Two reported networks, two directories of intermediate results:
#   elite     what the best individual scores, re-scored out of sample. THE
#             PERFORMANCE NUMBER.
#   centroid  what the mean of the population's WEIGHTS scores. For ES/NES that
#             IS the incumbent; for GA and DNS it measures whether the archive
#             has collapsed, and is NOT their performance.
#
# THE ARMS. `ga` is gaussian+truncation. Since 2026-09-15 the paper reports no
# novelty arm: `dns_gaussian` (gaussian+novelty) and the Iso+LineDD pair
# (`ga_isoline`, `dns`) stay on disk and are drawn only when ARMS names them.
# `es` and `nes` are one method at two settings: since 2026-09-15 each family
# reports ONE of them, the one with the higher Cum. elite (ES_ARM, below).
#
# THE CELLS. The observation-noise family reports MountainCar at sigma 0.1
# (sigma 1.0 is 7-15x its velocity range). The other two families draw no
# observation offset at all: `_sigma1.0` in their cell names is a NAME the
# analysis scripts split the environment out of, not a setting -- what a run
# is, is `task_type` in its own config.
# ============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

FAMILY="${1:?usage: $0 noise|freq50|freq400|physics|actions|noise_2task|physics_2task|minigrid|kinetix|ant|ant_2task|ant_warmup|ant_2task_warmup|ant_speed|ant_speed_warmup|cheetah_noise05_t10|cheetah_noise025|ant_noise025|ant_speed24|ant_speed24_warmup|cheetah_friction|cheetah_action|ant_action|ant_noise025_t10|ant_friction_2task|cheetah_friction_2task}"
PY=.venv/bin/python
PROJECT=projects/iclr_2027
ARMS="${ARMS:-ga es nes ppo trac redo cchain pbt pbt2}"
STEPS="${STEPS:-verify evaluate diverge dormancy plastic plot generalist}"

# Per family: the suite (the paper directory), the tree, the reported cells,
# the solved threshold the generalist table classifies at (empty = the
# registry's, which every gymnax env has), and the plasticity rows. The
# gymnax trainers log dormancy, churn, weight RMS and NTK rank per record, so
# their `curvature` row is the live NTK column; the shared runners log none of
# those for the NE arms, so on MiniGrid every row comes from the checkpoint
# pass and `curvature` is left out (its Fisher-rank version is in the JSON).
SUITE=gymnax
# The output directory under paper/$SUITE. Empty = the family name; the gymnax
# families nest the schedule under the perturbation (noise/10task, noise/2task).
DEST=""
THRESHOLD=""
PLASTIC_ROWS="dormancy_ckpt dormant_age action_shift_ckpt weight curvature step"
# Panels per row in the STATIONARY lineplot. Empty = one row, which is what
# three gymnax cells or two MiniGrid rooms want; twenty Kinetix levels wrap.
NONCONT_NCOLS=""
# The tree the STATIONARY phase is read from when ROOT carries none of its
# own: FT, the stationary figures and the stationary evaluation. Empty = ROOT.
REF_ROOT=""
case "$FAMILY" in
    noise)
        ROOT=$PROJECT/runs_centroid/gymnax
        DEST=noise/10task
        CELLS="CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma0.1" ;;
    freq50|freq400)
        # APPENDIX E, the switch interval (app:frequency). The same cells,
        # arms and 4000-generation budget as `noise`; only the sub-task
        # interval differs -- 50 or 400 generations against that family's 200
        # -- so the three are read side by side. The sub-task draw is seeded
        # by the trial alone and nests, so all three see the same 10 sub-tasks.
        #
        # The STATIONARY phase does not depend on the interval (there are no
        # switches in it), so both families borrow `noise`'s runs through
        # REF_ROOT rather than training a second copy: FT and the stationary
        # figures come from runs_centroid.
        #
        # freq400 is 10 phases of 400, so every sub-task is seen ONCE and
        # nothing is revisited: its F is final-agent forgetting, and the
        # switch-forgetting reading the 2-sub-task families use does not
        # apply. See the header of scripts/train/queue_iclr_frequency.sh.
        # The MATCHED tree (scripts/analysis/build_freq_matched.py): the
        # design's seeds only, as symlinks, so every arm is compared over
        # the same sub-task draws. runs_freq/ itself has extra seeds for
        # some arms and would mix trial sets.
        ROOT=$PROJECT/runs_freq_matched/interval${FAMILY#freq}/gymnax
        DEST=noise/interval${FAMILY#freq}
        REF_ROOT=$PROJECT/runs_centroid/gymnax
        CELLS="CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma0.1" ;;
    physics)
        ROOT=$PROJECT/runs_param_centroid/gymnax
        DEST=physics/10task
        CELLS="CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma1.0" ;;
    actions)
        # Two sub-tasks by construction: the stock action order and its
        # reverse, alternating over 20 phases.
        ROOT=$PROJECT/runs_actions/gymnax
        DEST=actions/2task
        CELLS="CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma1.0" ;;
    noise_2task)
        # Two sub-tasks under the observation offset (task_period=2): the clean
        # environment and ONE fixed offset, alternating over the same 20 phases
        # and 200-generation boundary as the ten-task family. Smaller offsets
        # than that family (0.5 / 0.5 / 0.05) so that a generalist over both
        # is attainable; the cell names carry the offset actually run.
        ROOT=$PROJECT/runs_noise_2task/gymnax
        DEST=noise/2task
        CELLS="CartPole_v1_sigma0.5 Acrobot_v1_sigma0.5 MountainCar_v0_sigma0.05" ;;
    physics_2task)
        # Two sub-tasks under the body rescaling: the default body and ONE
        # pinned multiplier (--param_range lo hi with lo == hi): CartPole pole
        # length x3, Acrobot link mass x1.15, MountainCar gravity x1.5, chosen
        # so a specialist has a reason to adapt but one policy can still cover
        # both. Alternating over 20 phases. _sigma1.0 is a name here, as in
        # physics. Acrobot was first run at x1.5, where no NE arm ever found a
        # generalist; that cell is kept in runs_param_2task/_acrobot_mass1.5.
        ROOT=$PROJECT/runs_param_2task/gymnax
        DEST=physics/2task
        CELLS="CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma1.0" ;;
    minigrid)
        # EmptyRandom-8x8 / EmptyRandom-16x16 alternating over 20 phases; the
        # stationary cells are one room each. 0.8: the 16x16 specialist scores
        # 0.95 / 0.97 on the two rooms and the 8x8 specialist 0.73 on the
        # 16x16 one, so 0.8 on BOTH is what a generalist must reach and
        # neither specialist does (source/studies/minigrid/settings.py).
        SUITE=minigrid
        ROOT=$PROJECT/runs_centroid/minigrid
        CELLS="MiniGrid_8x8_16x16"
        THRESHOLD=0.8
        PLASTIC_ROWS="dormancy_ckpt dormant_age action_shift_ckpt weight step" ;;
    kinetix)
        # The twenty medium levels (source/envs/kinetix_levels.py). The
        # continual cell is the chain over all twenty, 200 generations each;
        # the stationary cells are one level each at the same 200 generations
        # (1.311e7 steps: one 128-step rollout a genome since 2026-09-13; the
        # 256-step, 3-rollout `runs_kinetix` is the nominal-budget reference and
        # is not reported), so the twenty stationary panels wrap five to a
        # row. 1.0: a Kinetix return crosses 1.0 exactly when the goal was
        # reached (the terminal bonus), the solved criterion the working runs
        # verified with. Same plasticity rows as MiniGrid (shared runners, no
        # live NTK column).
        # runs_kinetix_paper is a tree of SYMLINKS, no data: `ga` is the
        # no-crossover variant in runs_nocrossover (ga_focus_explore_nox; the
        # same 200 gen x 512 x 1 x 128 budget, 4 continual trials); `es` (plain
        # OpenES since 2026-09-24) and every other arm are
        # runs_kinetix_ep128_ev1's (5 trials). See its README.
        SUITE=kinetix
        ROOT=$PROJECT/runs_kinetix_paper/kinetix
        CELLS="Kinetix20"
        THRESHOLD=1.0
        NONCONT_NCOLS=5
        PLASTIC_ROWS="dormancy_ckpt dormant_age action_shift_ckpt weight step" ;;
    ant)
        # The brax ant on MJX (source/studies/mjx/settings.py), BOTH continual
        # families of one body in one directory, since they share the
        # stationary cell: `ant_noise` adds a fixed offset (sigma 2.0) to the
        # 27-dim observation, `ant_friction` multiplies the ground friction
        # (log-uniform in [0.05, 5.0] per trial), and `ant` is the stock body
        # that is sub-task 0 of either. Ten sub-tasks over 20 phases as on
        # gymnax; 4.9152e8 steps a trial (NE: 320 gen x 512 x 3 x 1000; RL:
        # 48000 updates x 512 x 20), boundaries every 2.4576e7. No solved
        # threshold exists for the body (the score is reported, nothing
        # invents a constant), so the generalist table is not drawn. The
        # action is continuous: BD and the action-shift row are the
        # normalised distance between the two actions rather than an argmax
        # disagreement (`normalized_action_distance`), and the calibrated
        # KL row does not exist. The continual tree carries no GA/DNS arms
        # yet (the GA is being tuned on this body); the script names the gap.
        # Since 2026-09-15 the paper draws the friction cell alone: the sigma
        # 2.0 offset cell is superseded by `ant_noise025`.
        SUITE=mjx
        ROOT=$PROJECT/runs_ant/mjx
        CELLS="ant_friction"
        DEST=ant/physics/10task
        PLASTIC_ROWS="dormancy_ckpt dormant_age action_shift_ckpt weight step" ;;
    ant_speed24|ant_speed24_warmup)
        # The target-speed family at OVERLAPPING targets, 2 vs 4 m/s
        # (settings `ant_speed24`), otherwise `ant_speed` exactly: two
        # sub-tasks over 20 phases, or 2 m/s held for half the budget then 11
        # phases. Stationary reference runs_ant. Floor at 4 m/s measured at
        # launch (antspd24_smoke). No GA/DNS arms.
        SUITE=mjx
        ROOT=$PROJECT/runs_$FAMILY/mjx
        REF_ROOT=$PROJECT/runs_ant/mjx
        CELLS="ant_speed24"
        DEST=ant/speed_2v4/2task${FAMILY#ant_speed24}
        PLASTIC_ROWS="dormancy_ckpt dormant_age action_shift_ckpt weight step" ;;
    ant_speed|ant_speed_warmup)
        # The TARGET-SPEED family (source/studies/mjx/settings.py `ant_speed`):
        # a sub-task is the speed the reward tracks, 2.0 vs 8.0 m/s, the two
        # alternating over 20 phases (ant_speed) or 2.0 held for half the
        # budget then alternating over 11 phases (ant_speed_warmup), at the
        # ant's budget and boundaries. Sub-task 0 IS the stock body at 2 m/s,
        # so the stationary reference is runs_ant (REF_ROOT; make_lineplot
        # REF_ENV maps the cell to `ant`). The do-nothing floor is ~1514 on
        # BOTH targets (measured 2026-09-13, speed_conflict.py), not the
        # friction cell's 1470. No GA/DNS arms were run on this family.
        SUITE=mjx
        ROOT=$PROJECT/runs_$FAMILY/mjx
        REF_ROOT=$PROJECT/runs_ant/mjx
        CELLS="ant_speed"
        DEST=ant/speed_2v8/2task${FAMILY#ant_speed}
        PLASTIC_ROWS="dormancy_ckpt dormant_age action_shift_ckpt weight step" ;;
    ant_2task|ant_warmup|ant_2task_warmup)
        # The `ant` family with the SCHEDULE changed and nothing else: same
        # body, cells, arms, reward and 4.9152e8 steps a trial (NE 320 gen x
        # 512 x 3 x 1000; RL 48000 updates x 512 x 20).
        #   ant_2task         two sub-tasks, the stock body and ONE offset
        #                     (noise) or ONE friction multiplier (friction),
        #                     alternating over 20 phases of 16 gen / 2400
        #                     updates, as `ant`'s boundaries.
        #   ant_warmup        ten sub-tasks, but sub-task 0 is held for HALF
        #                     the budget (task_warmup 160 gen / 24000 updates)
        #                     before ten ordinary phases: 11 phases, 1..9 seen
        #                     once and sub-task 0 revisited last.
        #   ant_2task_warmup  the same warm-up, then the two alternating: 11
        #                     phases.
        # Under a warm-up the first phase is ten times the others, so the
        # switches are NOT multiples of task_interval: the figure scripts
        # read the edges from task_warmup (make_lineplot.phase_edges), and FT
        # scores the long phase against as long a stationary window. These
        # trees have no stationary phase of their own; it is runs_ant's,
        # which is this body at this budget (REF_ROOT).
        SUITE=mjx
        ROOT=$PROJECT/runs_$FAMILY/mjx
        REF_ROOT=$PROJECT/runs_ant/mjx
        CELLS="ant_noise ant_friction"
        PLASTIC_ROWS="dormancy_ckpt dormant_age action_shift_ckpt weight step" ;;
    cheetah_noise05_t10|cheetah_noise025)
        # The cheetah observation offset at the widths the campaign converged
        # to (handoff 2026-09-13, "the headline"): width 0.5 over TEN sub-tasks
        # (runs_mjx_noise05_t10) and width 0.25 over TWO (runs_mjx_noise025,
        # the replication); 20 phases, the cheetah budget and RL shape of
        # `cheetah_noise`. No GA/DNS arms were run on either. Stationary
        # reference: runs_mjx's `cheetah` cell (REF_ROOT; make_lineplot REF_ENV).
        SUITE=mjx
        case "$FAMILY" in
            cheetah_noise05_t10) ROOT=$PROJECT/runs_mjx_noise05_t10/mjx; DEST=cheetah/noise/10task ;;
            *)                   ROOT=$PROJECT/runs_mjx_noise025/mjx;    DEST=cheetah/noise/2task ;;
        esac
        REF_ROOT=$PROJECT/runs_mjx/mjx
        CELLS="cheetah_noise"
        PLASTIC_ROWS="dormancy_ckpt dormant_age action_shift_ckpt weight step" ;;
    ant_noise025)
        # The ant observation offset at width 0.25 over TWO sub-tasks, with
        # sub-task 0 held for half the budget (runs_ant_noise0.25): the one
        # ant family that clears the do-nothing floor (handoff 2026-09-13).
        # No GA/DNS arms. Stationary reference: runs_ant's `ant` cell.
        SUITE=mjx
        ROOT=$PROJECT/runs_ant_noise0.25/mjx
        DEST=ant/noise/2task_warmup
        REF_ROOT=$PROJECT/runs_ant/mjx
        CELLS="ant_noise"
        PLASTIC_ROWS="dormancy_ckpt dormant_age action_shift_ckpt weight step" ;;
    ant_noise025_t10)
        # The ant observation offset at width 0.25 over TEN sub-tasks, with the
        # warm-up of `ant_noise025` (sub-task 0 held for half the budget, then
        # ten phases: 11), so the two differ in the number of sub-tasks alone
        # (runs_ant_noise0.25_t10_wu). Of the three ten-sub-task ant offset
        # trees it is the one the NE arms learn most in: last-10% own-sub-task
        # score ES 1612 / NES 1470, against 1340 / 1407 at width 0.5
        # (runs_ant_noise0.5_t10) and 1487 / 1459 without the warm-up
        # (runs_ant_noise10_w0.25), the floor being ~1470. Its NE runs predate
        # the recorded whitening statistics (d09095e), so the post-hoc passes
        # re-measure them, ~8% low on the ant. No GA/DNS arms.
        SUITE=mjx
        ROOT=$PROJECT/runs_ant_noise0.25_t10_wu/mjx
        DEST=ant/noise/10task_warmup
        REF_ROOT=$PROJECT/runs_ant/mjx
        CELLS="ant_noise"
        PLASTIC_ROWS="dormancy_ckpt dormant_age action_shift_ckpt weight step" ;;
    ant_friction_2task)
        # The ant ground friction over TWO sub-tasks with the same warm-up as
        # `ant_noise025`: the friction cell of runs_ant_2task_warmup alone, into
        # the paper's physics directory (`ant_2task_warmup` draws both cells).
        SUITE=mjx
        ROOT=$PROJECT/runs_ant_2task_warmup/mjx
        DEST=ant/physics/2task_warmup
        REF_ROOT=$PROJECT/runs_ant/mjx
        CELLS="ant_friction"
        PLASTIC_ROWS="dormancy_ckpt dormant_age action_shift_ckpt weight step" ;;
    cheetah_friction_2task)
        # The cheetah ground friction over TWO sub-tasks alternating over 20
        # phases at the cheetah budget: runs_mjx_2task's friction cell. Its
        # generalist tables (threshold 2000) are written by hand into the same
        # directory; this family writes none (no solved threshold for mjx).
        SUITE=mjx
        ROOT=$PROJECT/runs_mjx_2task/mjx
        DEST=cheetah/physics/2task
        REF_ROOT=$PROJECT/runs_mjx/mjx
        CELLS="cheetah_friction"
        PLASTIC_ROWS="dormancy_ckpt dormant_age action_shift_ckpt weight step" ;;
    cheetah_noise|cheetah_friction)
        # The MJX cheetah (source/studies/mjx/settings.py), two continual
        # families on ONE tree: `cheetah_noise` adds a per-sub-task
        # observation offset at sigma 2.0, `cheetah_friction` multiplies the
        # ground friction; all eight arms in both. Ten distinct
        # sub-tasks visited twice over 20 phases, 4.915e8 steps per trial and
        # a boundary every 2.458e7 (NE: 320 gen x 512 pop x 3 evals x 1000
        # steps; RL: 6000 updates x 4096 envs x 20 steps), sub-task 0 the
        # stock body in both. The stationary cell is plain `cheetah` and
        # serves both families (make_lineplot.REF_ENV). No solved threshold
        # exists for the body (registry.threshold_for), so the stationary
        # Solved column is `--` and the generalist pass is skipped; the
        # continual curve IS the generalist score there. The NE runs whiten
        # their observations (`obs_norm`), which the post-hoc passes rebuild
        # from the run's own config.
        SUITE=mjx
        ROOT=$PROJECT/runs_mjx/mjx
        CELLS="$FAMILY"
        if [ "$FAMILY" = cheetah_friction ]; then DEST=cheetah/physics/10task; fi
        PLASTIC_ROWS="dormancy_ckpt dormant_age action_shift_ckpt weight step" ;;
    cheetah_action|ant_action)
        # Action reversal on the mjx bodies (task_mod `action`): TWO sub-tasks
        # alternating over 20 phases at the body's budget and boundaries (NE
        # 320 gen, 16 a phase). No stationary phase of their own: the body's
        # tree is the reference (REF_ROOT; make_lineplot REF_ENV). The cheetah
        # tree carries GA and DNS-gaussian, the ant tree does not; neither PBT.
        SUITE=mjx
        BODY=${FAMILY%_action}
        case "$BODY" in
            cheetah) ROOT=$PROJECT/runs_mjx_action/mjx; REF_ROOT=$PROJECT/runs_mjx/mjx ;;
            *)       ROOT=$PROJECT/runs_ant_action/mjx; REF_ROOT=$PROJECT/runs_ant/mjx ;;
        esac
        CELLS="$FAMILY"
        DEST=$BODY/actions/2task
        PLASTIC_ROWS="dormancy_ckpt dormant_age action_shift_ckpt weight step" ;;
    *)  echo "unknown family '$FAMILY'" >&2; exit 2 ;;
esac
ROOT="${ROOT_OVERRIDE:-$ROOT}"
CELLS="${CELLS_OVERRIDE:-$CELLS}"
NONCONT_ROOT=$ROOT
if [ ! -d "$ROOT/noncontinual" ] && [ -n "$REF_ROOT" ]; then
    NONCONT_ROOT=$REF_ROOT
    echo "=== $ROOT has no stationary phase: reading it from $NONCONT_ROOT" >&2
fi
[ -d "$ROOT/noncontinual" ] || [ -d "$ROOT/continual" ] || {
    echo "no run tree at $ROOT/{continual,noncontinual} -- nothing to draw for '$FAMILY'" >&2; exit 2; }
# A tree whose continual runs have not landed yet (Kinetix, 2026-09-11) still
# gets its stationary figures: the reference and the "can it solve the level
# at all" table. The continual passes are skipped and say so, once, here.
HAVE_CONT=1
[ -d "$ROOT/continual" ] || {
    HAVE_CONT=0
    echo "!!! $ROOT has no continual phase yet: drawing the stationary figures only" >&2; }

OUT=$PROJECT/paper/$SUITE/${DEST:-$FAMILY}
RESULTS=$OUT/results
mkdir -p "$RESULTS"/{elite,centroid}

has() { case " $STEPS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
# The arms that HAVE runs in a phase. An arm named in ARMS with no directory
# is a gap in the tree, not a typo: it is reported here, once, and left out of
# that phase's figures rather than failing every pass that takes --methods.
# The continual check is per REPORTED CELL, not per arm directory: on a tree
# that carries two families (the mjx bodies), `ppo/` exists for `cheetah_noise`
# and not for `cheetah_friction`, and a --methods list naming an arm with no
# runs in the cell is what makes the divergence pass fail on an empty glob.
present() {
    local phase=$1 out="" a c found
    for a in $ARMS; do
        found=0
        if [ "$phase" = continual ]; then
            for c in $CELLS; do [ -d "$ROOT/$phase/$a/$c" ] && found=1; done
        else
            [ -d "$NONCONT_ROOT/$phase/$a" ] && found=1
        fi
        if [ $found = 1 ]; then out="$out $a"
        else echo "!!! $phase: no runs for arm '$a' under $ROOT/$phase (cells: $CELLS)" >&2; fi
    done
    echo $out
}
CONT_ARMS=""
[ $HAVE_CONT = 1 ] && CONT_ARMS=$(present continual)
NONCONT_ARMS=$(present noncontinual)
echo "=== family: $FAMILY ($SUITE)   root: $ROOT   stationary: $NONCONT_ROOT"
echo "=== grid:   $CELLS"
echo "=== arms:   continual [$CONT_ARMS]  noncontinual [$NONCONT_ARMS]"
echo "=== steps:  $STEPS   ->  $OUT"

if has verify; then
    echo "=== verify ==="
    # Shared-runner runs: gymnax column names, in place, once. On a gymnax
    # tree this touches only the arms the shared runner made (pbt); the
    # gymnax trainers' own runs are recognised and left alone.
    $PY scripts/analysis/migrate_shared_runner_columns.py "$ROOT" | tail -1
    # The stationary tree can be another directory (the MJX families read the
    # body's own tree); its shared-runner arms need the same column names.
    [ "$NONCONT_ROOT" = "$ROOT" ] || $PY scripts/analysis/migrate_shared_runner_columns.py "$NONCONT_ROOT" | tail -1
    ok=1
    $PY scripts/verify_runs.py "$NONCONT_ROOT" --phase noncontinual --quiet || ok=0
    if [ $HAVE_CONT = 1 ]; then
        $PY scripts/verify_runs.py "$ROOT" --phase continual --quiet || ok=0
    fi
    if [ $ok = 0 ]; then
        if [ "${ALLOW_INCOMPLETE:-0}" = 1 ]; then
            echo "!!! tree is incomplete; drawing anyway because ALLOW_INCOMPLETE=1"
        else
            echo "!!! tree is incomplete; set ALLOW_INCOMPLETE=1 to draw it anyway" >&2
            exit 1
        fi
    fi
fi

# ONE ES-family arm a family. `es` and `nes` are one method at two settings, so
# every pass below draws whichever has the higher Cum. elite on this family's
# cells, as a scale-free margin summed over them (scripts/analysis/es_arm.py;
# the numbers go to $OUT/es_arm.json, which the cross-suite figures read). The
# other arm is dropped from the stationary reference too, so FT subtracts the
# kept arm's own run. Training curves only: CPU, seconds, and after verify so
# the shared-runner columns are migrated. ES_ARM=es|nes forces one arm and
# ES_ARM=both draws both.
ES_ARM="${ES_ARM:-best}"
without() { local x=$1 o="" a; shift; for a in "$@"; do [ "$a" = "$x" ] || o="$o $a"; done; echo $o; }
case " $CONT_ARMS $NONCONT_ARMS " in *" es "*) have_es=1 ;; *) have_es=0 ;; esac
case " $CONT_ARMS $NONCONT_ARMS " in *" nes "*) have_nes=1 ;; *) have_nes=0 ;; esac
if [ "$ES_ARM" = both ] || [ $have_es = 0 ] || [ $have_nes = 0 ]; then
    echo '{"kept": null, "criterion": "both drawn: ES_ARM=both, or one arm has no runs"}' > "$OUT/es_arm.json"
else
    if [ "$ES_ARM" = best ]; then
        if [ $HAVE_CONT = 1 ]; then
            ES_ARM=$($PY scripts/analysis/es_arm.py "$ROOT" --phase continual --cells $CELLS --out "$OUT/es_arm.json" | tail -1)
        else
            ES_ARM=$($PY scripts/analysis/es_arm.py "$NONCONT_ROOT" --phase noncontinual --out "$OUT/es_arm.json" | tail -1)
        fi
    else
        echo "{\"kept\": \"$ES_ARM\", \"criterion\": \"forced by ES_ARM\"}" > "$OUT/es_arm.json"
    fi
    if [ -z "$ES_ARM" ]; then
        echo "!!! es_arm.py found no ES/NES curves to compare: drawing both" >&2
    else
        drop=$([ "$ES_ARM" = es ] && echo nes || echo es)
        CONT_ARMS=$(without $drop $CONT_ARMS)
        NONCONT_ARMS=$(without $drop $NONCONT_ARMS)
        echo "=== ES-family arm: $ES_ARM ($drop not reported)   continual [$CONT_ARMS]  noncontinual [$NONCONT_ARMS]"
    fi
fi

if has evaluate; then
    echo "=== post-hoc evaluation (ZT) ==="
    # Tree-wide and idempotent: skips a run whose evaluation.json exists.
    if [ $HAVE_CONT = 1 ]; then
        $PY -m source.studies.evaluate_continual \
            --root "$ROOT/continual" --episodes 100 --gpus "${GPUS:-0}"
    fi
    # The stationary tree too: its agents on the room they never left is
    # the reference the generalist table is read against, and the pass is
    # idempotent. The gymnax stationary runs save no checkpoints.npz (the
    # trainers write it under continual only), so there the evaluator finds
    # nothing and exits non-zero; that is a no-op, not a failure, and must
    # not abort the passes after it.
    $PY -m source.studies.evaluate_continual \
        --root "$NONCONT_ROOT/noncontinual" --episodes 100 --gpus "${GPUS:-0}" \
        || echo "!!! no stationary checkpoints to evaluate under $NONCONT_ROOT/noncontinual (gymnax saves none); ZT/generalist read the continual tree only"
fi

if has diverge && [ $HAVE_CONT = 1 ]; then
    echo "=== behavioural divergence (F, BD) ==="
    # F and BD are a property of the saved sub-task agents, so they are
    # measured once per REPORTED agent: the centroid table must not carry the
    # elite's forgetting. `--agent_source elite` resolves to `finalgen`; the
    # single-policy RL arms carry `final` only, which is both.
    for agent in elite centroid; do
        src=finalgen; [ "$agent" = centroid ] && src=centroid
        $PY scripts/analysis/behavioural_divergence.py \
            --runs_root "$ROOT" --methods $CONT_ARMS --envs $CELLS \
            --num_tasks 20 --episodes 20 --num_states 2000 --gpus "${GPUS:-0}" \
            --agent_source "$src" --results_dir "$RESULTS/$agent"
    done
fi

if has dormancy && [ $HAVE_CONT = 1 ]; then
    echo "=== per-unit dormancy checkpoints (persistence, step) ==="
    # JAX_PLATFORMS=cpu: the nets are 386 parameters and the GPU buys nothing.
    # CUDA_VISIBLE_DEVICES="" is NOT enough -- jax still tries the cuda backend.
    # Except on the mjx bodies: the dormancy probe is a random walk through
    # the physics (`mjx.random_policy_observations`), one per distinct
    # sub-task row, and an MJX rollout on the CPU is minutes where the GPU
    # takes seconds. Same card as the other GPU passes, no preallocation, so
    # it shares the card the way they do.
    DORM_ENV="JAX_PLATFORMS=cpu"
    [ "$SUITE" = mjx ] && DORM_ENV="CUDA_VISIBLE_DEVICES=${GPUS:-0} XLA_PYTHON_CLIENT_PREALLOCATE=false"
    # Once per reported network: `--agent` picks the saved vector out of
    # checkpoints.npz and is passed to the figure too, so the curve rows read
    # `ne_centroid_*` under centroid instead of describing the elite twice.
    # One process PER ARM, then a merge: the pass keeps every jitted probe
    # program alive, and nine arms x twenty bodies on the physics family
    # exhausted the CPU JIT (LLVM "Cannot allocate memory", then a segfault)
    # where any eight arms passed alone (2026-09-13). Same split as the
    # divergence pass uses on the MJX bodies.
    for agent in elite centroid; do
        for arm in $CONT_ARMS; do
            env $DORM_ENV $PY scripts/analysis/plasticity_checkpoints.py \
                --runs_root "$ROOT" --phase continual --cells $CELLS \
                --agent "$agent" --methods $arm --out "$RESULTS/$agent/shards/$arm"
        done
        $PY scripts/analysis/merge_plasticity_checkpoints.py \
            "$RESULTS/$agent"/shards/*/plasticity_checkpoints.json \
            --out "$RESULTS/$agent/plasticity_checkpoints.json"
    done
    # Persistence and dormancy age for plot_plasticity_overview.py: the
    # centroid again on ONE fixed batch pooled over the distinct sub-tasks,
    # counting SILENT units only (mean |output| ~0) on every activation.
    # The matched probe moves with the observation offset, which redraws the
    # dormant set at every switch whatever the network does.
    for arm in $CONT_ARMS; do
        env $DORM_ENV $PY scripts/analysis/plasticity_checkpoints.py \
            --runs_root "$ROOT" --phase continual --cells $CELLS \
            --agent centroid --probe pooled --criterion magnitude --no-curvature --methods $arm \
            --out "$RESULTS/centroid_pooled/shards/$arm"
    done
    $PY scripts/analysis/merge_plasticity_checkpoints.py \
        "$RESULTS/centroid_pooled"/shards/*/plasticity_checkpoints.json \
        --out "$RESULTS/centroid_pooled/plasticity_checkpoints.json"
fi

if has plastic && [ $HAVE_CONT = 1 ]; then
    echo "=== plasticity figures ==="
    for agent in elite centroid; do
        JAX_PLATFORMS=cpu $PY scripts/make_plasticity_figure.py "$ROOT" \
            --phase continual --cells $CELLS \
            --checkpoints "$RESULTS/$agent" --agent "$agent" --methods $CONT_ARMS \
            --rows $PLASTIC_ROWS \
            --out "$OUT/plasticity_$agent"
    done
    # Population width, NE arms only: genomic diversity, fitness s.d. and the
    # pooled weight RMS per generation. These are population statistics, so
    # `--agent` cannot move them and they are drawn once. Under the
    # action-reversal family they are the result: ES/NES halt at a fitness
    # s.d. of exactly 0, and GA/DNS cross on width alone. The gymnax trainers
    # alone log them (`bd_*`, `weight_rms`); the shared runners keep no
    # population statistics per generation, so there is no such figure for
    # MiniGrid.
    if [ "$SUITE" = gymnax ]; then
        JAX_PLATFORMS=cpu $PY scripts/make_plasticity_figure.py "$ROOT" \
            --phase continual --cells $CELLS --methods $CONT_ARMS \
            --rows genomic_diversity fitness_std weight_pop \
            --out "$OUT/diversity"
    fi
fi

if has plot; then
    echo "=== lineplots and tables ==="
    for pair in "elite_eval elite" "centroid centroid"; do
        set -- $pair
        [ $HAVE_CONT = 1 ] || break
        $PY scripts/make_lineplot.py "$ROOT" \
            --phase continual --cells $CELLS --metric "$1" --methods $CONT_ARMS \
            --results-dir "$RESULTS/$2" --ref-root "$NONCONT_ROOT" \
            --out "$OUT/continual_$2"
        # The same table as a figure: mean over seeds with a 95% bootstrap
        # CI per (metric, environment), the main-text replacement for the
        # nineteen-column table, which stays beside the lineplot for the
        # appendix. Same loader, same runs, same estimator as the lineplot's
        # band -- the per-seed outcomes here are bimodal, so a median or an
        # IQR would move the RL arms a full ceiling on CartPole.
        $PY scripts/make_metrics_figure.py "$ROOT" \
            --phase continual --cells $CELLS --metric "$1" --methods $CONT_ARMS \
            --results-dir "$RESULTS/$2" --ref-root "$NONCONT_ROOT" \
            --out "$OUT/metrics_$2"
    done
    # The stationary reference at BOTH metrics. The two differ for GA and
    # DNS: under `centroid` the stationary GA archive scores ~400 on CartPole
    # against ~490 for its elite, because truncation selection never collapses
    # the archive and the mean of its weights is not an individual. That is
    # also the baseline the centroid table's FT column subtracts, which is
    # why GA's centroid FT comes out positive there while its elite FT
    # equals PPO's -- read the two references side by side, or FT under
    # `centroid` looks like transfer when it is archive width.
    # No --cells: the noncontinual cells carry no sigma in their names. ZT
    # there is the post-hoc re-evaluation of the saved agents on their own
    # level (`evaluate` runs over the stationary tree too).
    for pair in "elite_eval elite" "centroid centroid"; do
        set -- $pair
        # --threshold feeds the stationary table's Solved column (Final >=
        # threshold, k/n trials); empty = the run's own or the registry's.
        $PY scripts/make_lineplot.py "$NONCONT_ROOT" \
            --phase noncontinual --metric "$1" --methods $NONCONT_ARMS \
            ${NONCONT_NCOLS:+--ncols $NONCONT_NCOLS} \
            ${THRESHOLD:+--threshold $THRESHOLD} \
            --out "$OUT/noncontinual_$2"
    done
fi

if has generalist && [ $HAVE_CONT = 1 ] && [ "$SUITE" = mjx ] && [ -z "$THRESHOLD" ]; then
    echo "!!! $SUITE has no solved threshold: the generalist table is not drawn" >&2
elif has generalist && [ $HAVE_CONT = 1 ] && [ "$SUITE" != gymnax ] && [ -z "$THRESHOLD" ]; then
    # generalist_checkpoints.py classifies against a solved threshold and
    # exits when a body has none in the registry (the mjx bodies). The
    # per-sub-task scores it would classify are still in evaluation.json.
    echo "!!! generalist: no solved threshold for suite '$SUITE'; pass skipped" >&2
elif has generalist && [ $HAVE_CONT = 1 ]; then
    echo "=== generalist / switching specialist ==="
    # From the `evaluate` pass: the agent saved at the end of each phase,
    # scored on that phase's sub-task and on the previous one. Once per
    # reported agent, like every other post-hoc column.
    for agent in elite centroid; do
        $PY scripts/analysis/generalist_checkpoints.py "$ROOT" \
            --phase continual --cells $CELLS --agent "$agent" \
            --methods $CONT_ARMS ${THRESHOLD:+--threshold $THRESHOLD} \
            --out "$OUT/generalist_checkpoints_$agent"
    done
fi

echo "=== done: $OUT ==="
