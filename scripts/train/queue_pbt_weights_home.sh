#!/bin/bash
# ============================================================================
# PBT-PPO WITHOUT EXPLORE (pbt_weights, --pbt_mode weights_only) on the gymnax
# cells the paper draws, at home (2026-09-25), into the paper's own trees
# beside pbt / pbt2, with the settings of scripts/train/cluster/submit_gymnax_pbt.sh
# (NUM_TASKS=20, TASK_INTERVAL=200, the family's period / sigma / task type).
# cluster runs the same cells through submit_pbt_weights_paper.sh; whichever
# finishes a trial first keeps it (ship_home.sh skips trials home has).
#
#   noise 10-task   runs_centroid      CartPole 1.0, Acrobot 1.0, MountainCar 0.1   period 10
#   actions 2-task  runs_actions       all three at sigma 1.0 (name only)            period 2
#   noise 2-task    runs_noise_2task   MountainCar 0.05 (CartPole / Acrobot came from
#                                      queue_pbt_modes_probe.sh, copied in)         period 2
#   physics 2-task  runs_param_2task   CartPole length x3, MountainCar gravity x1.5;
#                   runs_param_2task_acrobot1.15  Acrobot mass x1.15                period 2
#
# Four launchers with disjoint cards (launch.sh leases per launcher), two PBT
# jobs a card (~24 GB each).  bash scripts/train/queue_pbt_weights_home.sh
# ============================================================================
set -eu
cd "$(dirname "$0")/../.."
export NUM_TRIALS="${NUM_TRIALS:-10}" NUM_TASKS=20 TASK_INTERVAL=200
ARM="${ARMS:-pbt_weights}"
R=projects/iclr_2027
launch() {   # gpus project_root task_type period sigmas envs [extra env assignments...]
    local gpus=$1 root=$2 type=$3 period=$4 sigmas=$5 envs=$6; shift 6
    env "$@" GPUS="$gpus" PROJECT_ROOT="$root" TASK_TYPE="$type" TASK_PERIOD="$period" SIGMAS="$sigmas" ENVS="$envs" \
        LOG_DIR="logs/launch_pbt_weights_$(basename "$root")_${type}_${sigmas// /_}" \
        bash scripts/train/launch.sh gymnax_continual "$ARM"
}
( launch "0 0 1 1" $R/runs_centroid noise 10 1.0 "CartPole-v1 Acrobot-v1"
  launch "0 0 1 1" $R/runs_centroid noise 10 0.1 "MountainCar-v0" ) &
( launch "2 2 3 3" $R/runs_actions actions 2 1.0 "CartPole-v1 Acrobot-v1 MountainCar-v0" ) &
( launch "4 4 5 5" $R/runs_param_2task param 2 1.0 "CartPole-v1" PARAM_RANGE="3.0 3.0"
  launch "4 4 5 5" $R/runs_param_2task param 2 1.0 "MountainCar-v0" PARAM_RANGE="1.5 1.5" ) &
( launch "6 6 7 7" $R/runs_noise_2task noise 2 0.05 "MountainCar-v0"
  launch "6 6 7 7" $R/runs_param_2task_acrobot1.15 param 2 1.0 "Acrobot-v1" PARAM_RANGE="1.15 1.15" ) &
wait
echo "queue_pbt_weights_home done $(date -Is)"
