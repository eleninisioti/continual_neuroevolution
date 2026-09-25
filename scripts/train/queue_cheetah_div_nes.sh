#!/bin/bash
# ============================================================================
# The HalfCheetah NES trials population_diversity_continual (paper/visuals/main)
# still lacks: submit_population_diversity.sh's CLUSTER re-runs delivered no
# cheetah_noise NES and only trials 5 and 8 of cheetah_action NES. Same flags
# as that script's mjx_sched.sbatch jobs, into the same tree names, at home:
#
#   noise    runs_div_noise05_t10   nes  10 sub-tasks, offset 0.5   trials 1-10
#   actions  runs_div_action        nes  2 sub-tasks                 all but 5, 8
#
# Waits until every GPU in WAIT_GPUS is idle, then runs two launchers on
# disjoint GPU sets (launch.sh leases are per launcher), two NE jobs a GPU.
#
#   bash scripts/train/queue_cheetah_div_nes.sh
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
WAIT_GPUS="${WAIT_GPUS:-3 4 5 6 7}"
R=projects/iclr_2027
LOG=logs/cheetah_div_nes; mkdir -p "$LOG"
NE="--obs_norm --track_plasticity"

busy() {
    for g in $WAIT_GPUS; do
        m=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
        [ "$m" -gt 100 ] && return 0
    done
    return 1
}
while busy; do sleep 60; done
echo "GPUs [$WAIT_GPUS] idle at $(date -Is)"

printf 'nes\tcheetah_action\t%s\n' 1 2 3 4 6 7 9 10 > "$LOG/actions_jobs.tsv"

GPUS="3 3 4 4 5 5" NUM_TRIALS=10 ENVS=cheetah_noise PROJECT_ROOT=$R/runs_div_noise05_t10 \
MJX_EXTRA="$NE --num_tasks 10 --noise_range 0.5" LOG_DIR=$LOG/noise \
    bash scripts/train/launch.sh cheetah_continual nes &
GPUS="6 6 7 7" JOBLIST=$LOG/actions_jobs.tsv PROJECT_ROOT=$R/runs_div_action \
MJX_EXTRA="$NE --num_tasks 2" LOG_DIR=$LOG/actions \
    bash scripts/train/launch.sh cheetah_continual nes &
wait
echo "done $(date -Is)"
