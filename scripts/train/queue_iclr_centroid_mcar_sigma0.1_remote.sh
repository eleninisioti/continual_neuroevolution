#!/bin/bash
# ============================================================================
# The SECOND-MACHINE half of the MountainCar sigma=0.1 cell.
#
# WHY A SPLIT. queue_iclr_centroid_mcar_sigma0.1.sh is eight arms x 10 trials
# on MountainCar, and on two cards at four jobs each that is ~15 h of wall
# clock -- the RL arms dominate it (median per trial, measured on the sigma=1.0
# cell of this same tree: nes 154 min, cchain 137, trac 132, redo 101, ppo 71,
# against ga 65 / es 62 / dns_gaussian ~60). The originating machine is already
# most of the way through `ga` and is running `dns_gaussian`; those two are the
# cheap arms and they finish there. This script takes the six that do not.
#
#     stays home   ga, dns_gaussian          ~4 h remaining
#     comes here   es nes ppo trac redo cchain   60 jobs, ~13 h on 2 cards
#
# Override with ARMS=... if the split moves.
#
# EVERY SETTING IS COPIED FROM queue_iclr_centroid_mcar_sigma0.1.sh, which
# copied them from queue_iclr_centroid.sh. That is not redundancy, it is the
# one rule this tree keeps breaking: a queue that sets only PROJECT_ROOT and
# SIGMAS inherits block_gymnax_continual's DEFAULTS (NUM_TASKS=10,
# TASK_PERIOD=0) and writes half-budget runs over sub-tasks that are never
# revisited, into a cell whose other arms are at the full budget. That has now
# happened twice -- `trac` via launch_centroid_trac_retry, and `dns_gaussian`
# via a launcher that started before its queue script was fixed. Both had to be
# thrown away. CLAUDE.md rule (c): every arm in a cell sees the same number of
# steps and meets its boundaries at the same step.
#
# NO NONCONTINUAL BLOCK, for the reason the sigma=0.1 queue gives: sub-task 0
# is the unperturbed environment under every sigma, so the stationary phase does
# not depend on one and the FT reference is already on disk for all eight arms.
#
# SAFE TO RE-RUN, AND SAFE TO OVERLAP WITH THE HOME MACHINE: run_condition
# skips any (method, env, trial) whose training_metrics.json exists, and the
# two halves write disjoint arm directories.
#
#   # on the second machine, from the repo root
#   bash scripts/train/queue_iclr_centroid_mcar_sigma0.1_remote.sh
#
#   # then bring the results home (from the second machine)
#   rsync -av --ignore-existing \
#       projects/iclr_2027/runs_centroid/gymnax/continual/ \
#       <home>:<repo>/projects/iclr_2027/runs_centroid/gymnax/continual/
#
# Env: ARMS, GPUS, NUM_TRIALS (default 10), SIGMAS (default 0.1),
#      JOBS_PER_GPU (default 4).
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ ! -f "$REPO_ROOT/pyproject.toml" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
cd "$REPO_ROOT"

ARMS="${ARMS:-es nes ppo trac redo cchain}"
ROOT=projects/iclr_2027/runs_centroid
TRIALS="${NUM_TRIALS:-10}"
SIGMA="${SIGMAS:-0.1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"

# launch.sh leases one token per entry in GPUS, so repeating an index is how it
# is told to oversubscribe a card. A gymnax job is ~600 MiB; four fit with room
# to spare on anything with 8 GB. Unlike the RL reproduction runs these are new
# runs where only the statistics matter, so sharing a card is fine -- see the
# bit-reproducibility note at the top of launch.sh.
if [ -z "${GPUS:-}" ]; then
    VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')
    VISIBLE=${VISIBLE:-0}
    GPUS=""
    for g in $VISIBLE; do
        for _ in $(seq 1 "$JOBS_PER_GPU"); do GPUS="$GPUS $g"; done
    done
    GPUS="${GPUS# }"
fi
export GPUS

echo "=== $(date -Is) MountainCar sigma $SIGMA continual block (remote half) ==="
echo "    arms  : $ARMS"
echo "    gpus  : $GPUS"
echo "    cell  : $ROOT/gymnax/continual/<arm>/MountainCar_v0_sigma$SIGMA"

NUM_TRIALS="$TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  ENVS="MountainCar-v0" \
  NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 SIGMAS="$SIGMA" \
  LOG_DIR="logs/launch_centroid_mcar_sigma${SIGMA}_remote" \
  bash scripts/train/launch.sh gymnax_continual $ARMS

echo "=== $(date -Is) done ==="
echo "failures:"
grep -hv 'exit=0' "logs/launch_centroid_mcar_sigma${SIGMA}_remote/status.tsv" 2>/dev/null || echo "  none"
