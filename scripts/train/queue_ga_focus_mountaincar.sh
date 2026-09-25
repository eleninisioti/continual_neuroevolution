#!/bin/bash
# Queue every `ga_focus_explore` MountainCar run (scripts/train/ga_focus_mountaincar.py):
# the stationary cell and the five gymnax continual families, 10 trials each.
# Finished runs are skipped, so the queue can be re-run after an interruption.
#
#   GPUS="5 6 7" PER_GPU=4 bash scripts/train/queue_ga_focus_mountaincar.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
GPUS=(${GPUS:-5 6 7})
PER_GPU=${PER_GPU:-4}
FAMILIES="${FAMILIES:-stationary noise_10task physics_10task actions_2task noise_2task physics_2task}"
TRIALS="${TRIALS:-1 2 3 4 5 6 7 8 9 10}"
LOG_DIR=projects/iclr_2027/runs_ga_focus_mountaincar/logs
mkdir -p "$LOG_DIR"

i=0
for family in $FAMILIES; do
    for trial in $TRIALS; do
        echo "$family $trial ${GPUS[$(( i % ${#GPUS[@]} ))]}"
        i=$(( i + 1 ))
    done
done | xargs -P $(( ${#GPUS[@]} * PER_GPU )) -L 1 bash -c \
    'nice -n 5 .venv/bin/python scripts/train/ga_focus_mountaincar.py "$0" "$1" "$2" \
        > '"$LOG_DIR"'/"$0"_t"$1".log 2>&1 && echo "done $0 trial $1" || echo "FAILED $0 trial $1"'
echo "queue finished"
