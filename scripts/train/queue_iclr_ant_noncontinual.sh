#!/bin/bash
# ============================================================================
# THE ANT STATIONARY BLOCK. Run this FIRST, on the second server.
#
# 80 trials: 8 arms x 1 cell x 10 trials, every one a stationary run of the
# full budget on the healthy ant. It is the control the continual block is read
# against -- a dormancy, rank or retention number only says something about
# task CHANGES if the same measurement on a run of the same length with no
# change does something different -- and it is also what says whether an arm
# can walk at all before anything is asked about carrying ten grounds.
#
# THE GRID
#
#     cell    ant             the unperturbed ant throughout (schedule task0)
#     arms    ga es nes dns_gaussian ppo trac redo cchain
#     trials  1..10
#
# ONE STATIONARY CELL SERVES BOTH FAMILIES, and that is not a shortcut.
# Sub-task 0 is the unperturbed ant under either kind of perturbation (a zero
# observation offset, a x1.0 friction multiplier); the sub-task vectors are
# drawn off the TRIAL rather than the run's key stream, so a `task0` run never
# touches the difference. Running it twice would be the same run twice at ~1-3
# h a trial. It is filed under the friction family so its nine idle evaluation
# columns are zero-shot friction transfer. See
# source/studies/mjx/settings.py.
#
# `ga` and `dns_gaussian` are the GAUSSIAN pair -- identical mutation, so they
# differ in the selection rule alone. The Iso+LineDD arms (`ga_isoline`,
# `dns`) are the operator ablation, are not in the paper, and are not run.
# This is why the run does not go through `block_brax_noncontinual`: those
# trainers' DNS breeds with Iso+LineDD only and none of them saves a centroid.
#
# COMPUTE-MATCHED, and asserted rather than asserted-in-a-comment: every arm
# gets 4.9152e8 environment steps, as 320 generations x 512 x 3 x 1000 for the
# NE arms and 48,000 updates x 512 x 20 for the RL ones, cut into the same
# twenty phases. `source/studies/mjx/settings.py:check()` recomputes both sides
# at the start of EVERY trial and refuses to run if they have drifted apart.
# It is the paper's own ant per-phase budget: 16 generations at pop 512 x 3
# evals is what every NE arm in runs_repro2/brax/continual got per sub-task.
#
# NO ARM IS TOLD WHERE A PHASE BOUNDARY IS (CLAUDE.md (d)) -- and on this block
# there is nothing to tell: the ground and the observation never change. The
# phase grid still exists because the checkpoints sit on it, twenty per run,
# exactly where the continual block's do, so the two are compared point for
# point.
#
# WHY RE-RUN AN ANT THAT IS ALREADY ON DISK. runs_repro2's ant trees cannot
# answer the three things this study reports:
#   centroid  the old trainers never save the mean of the population's
#             weights, so an ant curve there can only be the elite -- the
#             defect that made the gymnax centroid figure measure the wrong
#             network until 2026-09-09.
#   the pair  the old DNS breeds with Iso+LineDD only, so a GA-vs-DNS gap
#             confounds the selection rule with the variation operator.
#   the curve the old reported number is the search's own selection statistic
#             and carries the winner's curse. Here every generation re-scores
#             the centroid and the population mean on FRESH keys, feeding
#             nothing back.
#
# USAGE
#   git pull                                    # needs the guards below to pass
#   DRY_RUN=1 bash scripts/train/queue_iclr_ant_noncontinual.sh
#   nohup bash scripts/train/queue_iclr_ant_noncontinual.sh \
#       > logs/ant_noncontinual.log 2>&1 &
#
#   # then, and only then, the continual half
#   nohup bash scripts/train/queue_iclr_ant_continual.sh \
#       > logs/ant_continual.log 2>&1 &
#
#   # when both are done, send the tree home
#   rsync -av --ignore-existing \
#       projects/iclr_2027/runs_mjx/ \
#       <home>:<repo>/projects/iclr_2027/runs_mjx/
#
# COST. The 2026-09-06 ant runs at this exact budget took ~1 h a trial for the
# NE arms and ~3 h for the RL ones, ONE JOB TO A CARD -- 512 x 3 MJX ant
# rollouts fill a GPU, which a 600 MiB gymnax job does not. 80 trials is about
# 160 job-hours, so roughly 20 h on 8 cards. JOBS_PER_GPU is therefore 1 by
# default; raise it only once `nvidia-smi` has shown you what one job actually
# costs on your cards, and expect the NE arms to tolerate it better than PPO.
# NUM_TRIALS=5 halves the wall clock and is the thing to cut if time is short.
#
# Env: GPUS, JOBS_PER_GPU (1), NUM_TRIALS (10), ROOT, ARMS, CELLS, DRY_RUN.
#
# DO NOT EDIT run_experiments.sh OR launch.sh IN PLACE WHILE THIS RUNS. bash
# reads a script incrementally, so an in-place rewrite under a live launcher
# corrupts its parse. Write a temp file and `mv` it.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ROOT="${ROOT:-projects/iclr_2027/runs_mjx}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
NUM_TRIALS="${NUM_TRIALS:-10}"
DRY_RUN="${DRY_RUN:-0}"
ARMS="${ARMS:-ga es nes dns_gaussian ppo trac redo cchain}"
CELLS="${CELLS:-ant}"
LOGS=logs/ant_noncontinual
mkdir -p "$LOGS"

# ------------------------------------------------------------ version guard
# The same check whose absence cost eight GA trials on 2026-09-09: a launcher
# started against a checkout whose trainer could not yet write the centroid,
# and Python reads its source at process START, so runs that FINISHED after
# the fix still had no centroid in them. Nothing recovers it post hoc.
if [ ! -f source/studies/mjx/cli.py ]; then
    echo "FATAL: this checkout has no ant study. Run: git pull" >&2
    exit 1
fi
for f in train_nes train_ppo; do
    if ! grep -q 'save_checkpoints' "source/studies/generalists/${f}.py"; then
        echo "FATAL: source/studies/generalists/${f}.py cannot write" >&2
        echo "       checkpoints.npz, so no centroid figure is possible." >&2
        echo "       This checkout is too old. Run: git pull" >&2
        exit 1
    fi
done
# The mjx suite has to know that an ant sub-task can be a friction multiplier
# AND that it can be an observation offset; a checkout that predates either
# would silently run one family as the other.
if ! grep -q 'def friction_multipliers' source/envs/mjx.py; then
    echo "FATAL: source/envs/mjx.py has no friction sub-tasks. Run: git pull" >&2
    exit 1
fi
if ! grep -q 'ant_noncontinual)' scripts/train/run_experiments.sh; then
    echo "FATAL: run_experiments.sh has no ant blocks. Run: git pull" >&2
    exit 1
fi

# A real import of the settings, which is also the compute-match assertion.
# Cheaper to fail here than 80 times in a row inside the launcher.
if ! .venv/bin/python source/studies/mjx/cli.py \
        --env ant --method ppo --output_dir /tmp/ant_preflight --dry_run; then
    echo "FATAL: preflight failed -- settings or environment are broken." >&2
    exit 1
fi
echo "guards OK"

echo "root       : $ROOT/mjx/noncontinual"
echo "cells      : $CELLS"
echo "arms       : $ARMS"
echo "trials     : 1..$NUM_TRIALS"
echo "jobs       : $(( $(echo $ARMS | wc -w) * $(echo $CELLS | wc -w) * NUM_TRIALS ))"

# ------------------------------------------------------------ GPU pool
# One token per entry, leased by launch.sh -- repeat a GPU to oversubscribe it.
if [ -z "${GPUS:-}" ]; then
    VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')
    VISIBLE=${VISIBLE:-0}
    GPUS=""
    for g in $VISIBLE; do
        for _ in $(seq 1 "$JOBS_PER_GPU"); do GPUS="$GPUS $g"; done
    done
fi
echo "gpus       : $GPUS"

if [ "$DRY_RUN" != "0" ]; then
    DRY_RUN=1 GPUS="$GPUS" NUM_TRIALS="$NUM_TRIALS" ENVS="$CELLS" \
      PROJECT_ROOT="$ROOT" LOG_DIR="$LOGS" \
      bash scripts/train/launch.sh ant_noncontinual $ARMS
    echo "=== DRY RUN -- nothing queued ==="
    exit 0
fi

echo "=== $(date -Is) starting ==="
GPUS="$GPUS" NUM_TRIALS="$NUM_TRIALS" ENVS="$CELLS" \
  PROJECT_ROOT="$ROOT" LOG_DIR="$LOGS" \
  bash scripts/train/launch.sh ant_noncontinual $ARMS
code=$?
echo "=== $(date -Is) done (exit=$code) ==="
echo "failures:"
grep -v 'exit=0' "$LOGS/status.tsv" 2>/dev/null || echo "  none"
