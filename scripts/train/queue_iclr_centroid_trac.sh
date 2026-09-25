#!/bin/bash
# ============================================================================
# The `trac` arm of projects/iclr_2027/runs_centroid, CONTINUAL phase only,
# retrained at the settings the rest of the cell uses.
#
# WHY IT EXISTS. `logs/launch_centroid_trac_retry` completed trac's missing
# trials without repeating the queue's settings, so it took
# block_gymnax_continual's DEFAULTS -- NUM_TASKS=10, TASK_PERIOD=0 -- and wrote
# ten trials at 1.536e9 env steps over ten never-revisited sub-tasks into cells
# whose other nine arms are at 3.072e9 over twenty sub-tasks seen twice:
#
#     CartPole    trials 4, 6, 10
#     Acrobot     trials 4, 6, 8
#     MountainCar trials 4, 7, 9, 10
#
# Two things are wrong with those runs, and only one of them is the budget.
# CLAUDE.md rule (c) is broken outright -- the arms in that cell no longer saw
# the same number of environment steps or met their boundaries at the same
# step. But even alone they would be unusable: with TASK_PERIOD=0 nothing is
# revisited, and forgetting is defined as the drop on a sub-task you return to.
# scripts/analysis/fix_centroid_budget_mismatch.sh deletes them.
#
# A RETRY MUST REPEAT EVERY SETTING OF THE QUEUE IT COMPLETES. That is the
# whole lesson, and it is why this is a file rather than a command line: the
# settings below are copied from queue_iclr_centroid.sh and live somewhere they
# can be diffed against it.
#
# CONTINUAL ONLY. trac's noncontinual phase is complete and was never affected
# -- NUM_TASKS and TASK_PERIOD do not reach block_gymnax_noncontinual.
#
# NOT GPUS 4 AND 5 BY DEFAULT. queue_iclr_centroid_mcar_sigma0.1.sh holds those
# two cards, and launch.sh leases GPUs per launcher: two launchers given
# overlapping sets will both believe they own the card. GPUS=... overrides.
#
# SAFE TO RE-RUN: run_condition skips a (method, env, trial) whose
# training_metrics.json exists -- which is exactly why the bad trials must be
# DELETED before this runs, or it will skip them and change nothing.
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ ! -f "$REPO_ROOT/pyproject.toml" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
cd "$REPO_ROOT"

ARMS="${ARMS:-trac}"
ROOT=projects/iclr_2027/runs_centroid
TRIALS="${NUM_TRIALS:-10}"
export GPUS="${GPUS:-0 0 1 1 2 2 3 3 6 6 7 7}"

echo "=== $(date -Is) $ARMS continual block -> $ROOT ==="
echo "    gpus : $GPUS"

NUM_TRIALS="$TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 SIGMAS=1.0 \
  LOG_DIR=logs/launch_centroid_trac_fixed \
  bash scripts/train/launch.sh gymnax_continual $ARMS

echo "=== $(date -Is) done ==="
echo "failures:"
grep -hv 'exit=0' logs/launch_centroid_trac_fixed/status.tsv 2>/dev/null || echo "  none"
