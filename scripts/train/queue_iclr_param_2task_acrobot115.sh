#!/bin/bash
# ============================================================================
# Two-sub-task physics family, Acrobot only, at link mass x1.15 instead of x1.5.
#
# runs_param_2task (trained on CLUSTER) pinned Acrobot's link masses at x1.5.
# There no NE arm ever holds a generalist and GA/ES sit on "neither" for about
# half their post-switch checkpoints: the shift is too large for one policy to
# cover. At x1.15 (the generalists study's value, chosen from zero-shot transfer:
# it already drops a stock specialist from -69 to -94) every arm found a
# generalist. Everything else is runs_param_2task's setting: 20 sub-tasks,
# period 2, 200 generations each, _sigma1.0 a name only, diversity tracked.
#
# Runs land in a SEPARATE tree so the x1.5 cell keeps feeding the figures until
# this finishes; then swap the cell into runs_param_2task (the x1.5 cell goes to
# runs_param_2task/_acrobot_mass1.5) and re-run finish_iclr.sh physics_2task.
#
#   bash scripts/train/queue_iclr_param_2task_acrobot115.sh
#   Env: ARMS, NUM_TRIALS (10), JOBS_PER_GPU (2), GPUS, DRY_RUN
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ ! -f "$REPO_ROOT/pyproject.toml" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
cd "$REPO_ROOT"

ARMS="${ARMS:-ga dns_gaussian es nes ppo trac redo cchain}"
ROOT=projects/iclr_2027/runs_param_2task_acrobot1.15
TRIALS="${NUM_TRIALS:-10}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"

# Repeat an index to put more than one job on a card (launch.sh leases one
# token per entry). A gymnax job is ~600 MiB.
if [ -z "${GPUS:-}" ]; then
    VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')
    GPUS=""
    for g in ${VISIBLE:-0}; do
        for _ in $(seq 1 "$JOBS_PER_GPU"); do GPUS="$GPUS $g"; done
    done
    GPUS="${GPUS# }"
fi
export GPUS

mkdir -p "$ROOT/gymnax"
echo "=== $(date -Is) gymnax physics 2-task, Acrobot link mass x1.15 ==="
echo "    root  : $ROOT"
echo "    arms  : $ARMS"
echo "    gpus  : $GPUS"
echo "    trials: 1..$TRIALS"

NUM_TRIALS="$TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  ENVS="Acrobot-v1" PARAM_NAME=mass PARAM_RANGE="1.15 1.15" \
  TASK_TYPE=param NUM_TASKS=20 TASK_PERIOD=2 TASK_INTERVAL=200 SIGMAS=1.0 \
  LOG_DIR=logs/launch_param_2task_acrobot115 \
  bash scripts/train/launch.sh gymnax_continual $ARMS

echo "=== $(date -Is) done ==="
echo "failures:"
grep -hv 'exit=0' logs/launch_param_2task_acrobot115/status.tsv 2>/dev/null || echo "  none"
