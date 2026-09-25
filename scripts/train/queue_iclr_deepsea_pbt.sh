#!/bin/bash
# ============================================================================
# PBT-PPO on DeepSea 12 (the action-map family), into probe_deepsea beside the
# arms already there (GA, ES, GA + Novelty, PPO, TRAC, ReDo, C-CHAIN;
# logs/launch_probe_deepsea*). The SAME block settings the RL launch of
# 2026-09-19 used, so the run is compute-matched and switches on the same
# phase grid: NE 512 x 3 evals x 12 steps x 4000 generations = 73,728,000
# environment steps, 20 phases of 36 updates, action map with period 10.
#
# PBT is the shared runner (source/studies/generalists/train_ppo.py
# --method pbt, N=8, exploit+explore), whose members are the DeepSea PPO in
# PPO_CONFIGS (CartPole's settings at gamma 0.99, the gymnax RL trainer's
# entry). It builds DeepSea through gymnax_classic.build_env, so the action
# maps are the ones every other arm saw, and it logs bd_genomic_diversity /
# bd_fitness_std / bd_behavioural_diversity, which is what the population
# diversity figure (plot_population_diversity.py) reads.
#
#   bash scripts/train/queue_iclr_deepsea_pbt.sh
#       GPUS="2 3 4 5 6 7"   (default; 0 and 1 were at 48/49 GB on 2026-09-20)
#       ARMS="pbt"           pbt2 for the N=2 population
#       NUM_TRIALS=10  DRY_RUN=1
#
# Cost: ~1 min per 18 updates under contention (smoke test, 2026-09-20), so
# ~20 min a trial; 10 trials on 6 cards is under an hour. Then
#   bash scripts/analysis/finish_deepsea_pbt.sh
# for the links, the forgetting pass and the figures.
# ============================================================================
set -eu
cd "$(dirname "$0")/../.."
export NUM_TASKS=20 TASK_INTERVAL=200 TASK_PERIOD=10 TASK_TYPE=actions SIGMAS=1.0 \
       NE_POP_SIZE=512 NE_NUM_EVALS=3 NE_EPISODE_LENGTH=12 NE_HIDDEN_DIMS="16 16" \
       NUM_TRIALS="${NUM_TRIALS:-10}" ENVS=DeepSea12-bsuite \
       PROJECT_ROOT=projects/iclr_2027/probe_deepsea \
       LOG_DIR="${LOG_DIR:-logs/launch_probe_deepsea_pbt}"
export GPUS="${GPUS:-2 3 4 5 6 7}"
exec bash scripts/train/launch.sh gymnax_continual ${ARMS:-pbt}
