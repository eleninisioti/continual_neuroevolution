#!/bin/bash
# ============================================================================
# THE MiniGrid STATIONARY BLOCK. Run this FIRST, on the second server.
#
# 160 trials: 8 arms x 2 rooms x 10 trials, every one a stationary run of the
# full budget. It is the control the continual block is read against -- a
# dormancy, rank or retention number only says something about task CHANGES if
# the same measurement on a run of the same length with no change does
# something different -- and it is also what says whether an arm can learn
# either room at all before anything is asked about carrying both.
#
# THE GRID
#
#     cells   MiniGrid_8x8      EmptyRandom-8x8 throughout
#             MiniGrid_16x16    EmptyRandom-16x16 throughout
#     arms    ga es nes dns_gaussian ppo trac redo cchain
#     trials  1..10
#
# `ga` and `dns_gaussian` are the GAUSSIAN pair -- identical mutation, so they
# differ in the selection rule alone. The Iso+LineDD arms (`ga_isoline`,
# `dns`) are the operator ablation, are not in the paper, and are not run.
#
# COMPUTE-MATCHED, and asserted rather than asserted-in-a-comment: every arm
# gets 6.291e9 environment steps, as 4000 generations x 512 x 3 x 1024 for the
# NE arms and 61,440 updates x 2048 x 50 for the RL ones, and both are cut
# into the same twenty phases. `source/studies/minigrid/settings.py:check()`
# recomputes both sides at the start of EVERY trial and refuses to run if they
# have drifted apart.
#
# NO ARM IS TOLD WHERE A PHASE BOUNDARY IS (CLAUDE.md (d)) -- and on this
# block there is nothing to tell: the room never changes. The phase grid still
# exists because the checkpoints sit on it, twenty per run, exactly where the
# continual block's do, so the two are compared point for point.
#
# USAGE
#   git pull                                    # needs the guard below to pass
#   DRY_RUN=1 bash scripts/train/queue_iclr_minigrid_noncontinual.sh
#   nohup bash scripts/train/queue_iclr_minigrid_noncontinual.sh \
#       > logs/minigrid_noncontinual.log 2>&1 &
#
#   # then, and only then, the continual half
#   nohup bash scripts/train/queue_iclr_minigrid_continual.sh \
#       > logs/minigrid_continual.log 2>&1 &
#
#   # when both are done, send the tree home
#   rsync -av --ignore-existing \
#       projects/iclr_2027/runs_centroid/minigrid/ \
#       <home>:<repo>/projects/iclr_2027/runs_centroid/minigrid/
#
# COST. The 2026-09-07/08 MiniGrid runs took 0.7-3.9 h a trial for the NE arms
# and 4.9 h for PPO, one job to a card. 160 trials is roughly 500 job-hours,
# so about 31 h on 8 cards at 2 jobs each. JOBS_PER_GPU is 2 rather than the
# gymnax blocks' 4 because a MiniGrid job is not a 600 MiB gymnax job: PPO
# carries 2048 environments and the NE arms a 512 x 1024-step scan. Raise it
# once `nvidia-smi` has shown you what one job actually costs on your cards.
# NUM_TRIALS=5 halves the wall clock and is the thing to cut if time is short.
#
# Env: GPUS, JOBS_PER_GPU (2), NUM_TRIALS (10), ROOT, ARMS, CELLS, DRY_RUN.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ROOT="${ROOT:-projects/iclr_2027/runs_centroid}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
NUM_TRIALS="${NUM_TRIALS:-10}"
DRY_RUN="${DRY_RUN:-0}"
ARMS="${ARMS:-ga es nes dns_gaussian ppo trac redo cchain}"
CELLS="${CELLS:-MiniGrid_8x8 MiniGrid_16x16}"
LOGS=logs/minigrid_noncontinual
mkdir -p "$LOGS"

# ------------------------------------------------------------ version guard
# The same check whose absence cost eight GA trials on 2026-09-09: a launcher
# started against a checkout whose trainer could not yet write the centroid,
# and Python reads its source at process START, so runs that FINISHED after
# the fix still had no centroid in them. Nothing recovers it post hoc.
if [ ! -f source/studies/minigrid/cli.py ]; then
    echo "FATAL: this checkout has no MiniGrid study. Run: git pull" >&2
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
if ! grep -q 'minigrid_noncontinual)' scripts/train/run_experiments.sh; then
    echo "FATAL: run_experiments.sh has no minigrid blocks. Run: git pull" >&2
    exit 1
fi

# A real import of the settings, which is also the compute-match assertion.
# Cheaper to fail here than 160 times in a row inside the launcher.
if ! .venv/bin/python source/studies/minigrid/cli.py \
        --env MiniGrid_8x8 --method ppo --output_dir /tmp/minigrid_preflight \
        --dry_run; then
    echo "FATAL: preflight failed -- settings or environment are broken." >&2
    exit 1
fi
echo "guards OK"

echo "root       : $ROOT/minigrid/noncontinual"
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
      bash scripts/train/launch.sh minigrid_noncontinual $ARMS
    echo "=== DRY RUN -- nothing queued ==="
    exit 0
fi

echo "=== $(date -Is) starting ==="
GPUS="$GPUS" NUM_TRIALS="$NUM_TRIALS" ENVS="$CELLS" \
  PROJECT_ROOT="$ROOT" LOG_DIR="$LOGS" \
  bash scripts/train/launch.sh minigrid_noncontinual $ARMS
code=$?
echo "=== $(date -Is) done (exit=$code) ==="
echo "failures:"
grep -v 'exit=0' "$LOGS/status.tsv" 2>/dev/null || echo "  none"
