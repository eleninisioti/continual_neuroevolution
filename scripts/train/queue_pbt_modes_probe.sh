#!/bin/bash
# ============================================================================
# PBT-PPO mode probe (2026-09-25): what gives PBT-PPO its wide first solution,
# selection (exploit) or hyperparameter exploration (explore)? Two arms on the
# paper's noise two-task cell (paper/gymnax/data/noise_2task: 20 phases of 200
# NE generations = 1500 PPO updates, period 2, sigma 0.5), CartPole and
# Acrobot, the settings of the paper's `pbt` arm (mode full):
#   pbt_weights   exploit only  (weights copied, hyperparameters fixed)
#   pbt_hp        explore only  (hyperparameters perturbed, no copying)
# Afterwards: the centroid width pass and the overlap probe on the new root,
# against the paper's pbt / ppo arms over the first tasks.
#
#   bash scripts/train/queue_pbt_modes_probe.sh        # GPUS two jobs a card
#   NUM_TRIALS=10 DRY_RUN=1 ...
# ============================================================================
set -eu
cd "$(dirname "$0")/../.."
export NUM_TASKS=20 TASK_PERIOD=2 TASK_INTERVAL=200 TASK_TYPE=noise SIGMAS=0.5 \
       NUM_TRIALS="${NUM_TRIALS:-10}" ENVS="${ENVS:-CartPole-v1 Acrobot-v1}" \
       PROJECT_ROOT=projects/iclr_2027/probe_pbt_modes \
       LOG_DIR=logs/launch_probe_pbt_modes
export GPUS="${GPUS:-0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7}"
exec bash scripts/train/launch.sh gymnax_continual ${ARMS:-pbt_weights pbt_hp}
