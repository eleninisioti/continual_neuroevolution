#!/bin/bash
# ============================================================================
# THE MiniGrid CONTINUAL BLOCK. Run this SECOND, after the stationary one.
#
# 80 trials: 8 arms x 1 cell x 10 trials. The cell is `MiniGrid_8x8_16x16` --
# EmptyRandom-8x8 and EmptyRandom-16x16 alternating every phase, twenty
# phases, so each room is visited ten times and a REVISIT is what measures
# retention. That is the same structure as the gymnax grid's NUM_TASKS=20
# TASK_PERIOD=10, which is why the two bodies can be read side by side.
#
# WHY SECOND. The stationary block answers "can this arm learn either room at
# all, on this budget?", and every number here is a comparison against it: an
# arm that never solves 16x16 alone has not forgotten it. Running this first
# risks 80 trials whose result cannot be interpreted. It is not a technical
# dependency -- the two write into disjoint directories and either order
# works -- it is that the second block's figures need the first block's.
#
# THE PAIR is nested: the 16x16 specialist solves both rooms (0.95 / 0.97 at
# 32 fixed seeds) and the 8x8 specialist still reaches 0.73 on the 16x16 one.
# So a generalist demonstrably EXISTS and "did switching cost you one?" is a
# question with a known-attainable answer. `source/envs/minigrid.py` records
# the probes that chose this pair over the key-and-door tasks, where every NE
# arm scores a flat zero and no comparison is possible at all.
#
# ARMS, BUDGET AND BOUNDARY INFORMATION are the stationary block's, verbatim:
# ga es nes dns_gaussian ppo trac redo cchain, the gaussian pair only, 6.291e9
# environment steps each with the match asserted at the start of every trial,
# and no arm told where a boundary is (CLAUDE.md (d)) -- C-CHAIN never resets
# its coefficient, and the GA and DNS re-evaluate their stored population every
# generation rather than only at a transition. `--oracle` builds the
# deliberately-informed control and nothing here passes it.
#
# USAGE
#   DRY_RUN=1 bash scripts/train/queue_iclr_minigrid_continual.sh
#   nohup bash scripts/train/queue_iclr_minigrid_continual.sh \
#       > logs/minigrid_continual.log 2>&1 &
#
#   rsync -av --ignore-existing \
#       projects/iclr_2027/runs_centroid/minigrid/ \
#       <home>:<repo>/projects/iclr_2027/runs_centroid/minigrid/
#
# COST. Half the stationary block: 80 trials, roughly 250 job-hours, about
# 16 h on 8 cards at 2 jobs each. See the other script's header for why
# JOBS_PER_GPU starts at 2 and what to check before raising it.
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
CELLS="${CELLS:-MiniGrid_8x8_16x16}"
LOGS=logs/minigrid_continual
mkdir -p "$LOGS"

# ------------------------------------------------------------ version guard
# See the stationary script: Python reads its source at process START, so a
# launcher begun against a checkout that cannot write `checkpoints.npz` gives
# runs with no centroid in them however long after the fix they finish, and
# nothing recovers it post hoc.
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
if ! grep -q 'minigrid_continual)' scripts/train/run_experiments.sh; then
    echo "FATAL: run_experiments.sh has no minigrid blocks. Run: git pull" >&2
    exit 1
fi
if ! .venv/bin/python source/studies/minigrid/cli.py \
        --env MiniGrid_8x8_16x16 --method ppo \
        --output_dir /tmp/minigrid_preflight --dry_run; then
    echo "FATAL: preflight failed -- settings or environment are broken." >&2
    exit 1
fi
echo "guards OK"

echo "root       : $ROOT/minigrid/continual"
echo "cells      : $CELLS"
echo "arms       : $ARMS"
echo "trials     : 1..$NUM_TRIALS"
echo "jobs       : $(( $(echo $ARMS | wc -w) * $(echo $CELLS | wc -w) * NUM_TRIALS ))"

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
      bash scripts/train/launch.sh minigrid_continual $ARMS
    echo "=== DRY RUN -- nothing queued ==="
    exit 0
fi

echo "=== $(date -Is) starting ==="
GPUS="$GPUS" NUM_TRIALS="$NUM_TRIALS" ENVS="$CELLS" \
  PROJECT_ROOT="$ROOT" LOG_DIR="$LOGS" \
  bash scripts/train/launch.sh minigrid_continual $ARMS
code=$?
echo "=== $(date -Is) done (exit=$code) ==="
echo "failures:"
grep -v 'exit=0' "$LOGS/status.tsv" 2>/dev/null || echo "  none"
