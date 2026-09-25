#!/bin/bash
# ============================================================================
# Appendix: the effect of HOW OFTEN the sub-task changes (gymnax, obs. noise).
#
# The reported gymnax noise cell (runs_centroid) switches every 200
# generations: 20 phases x 200 = 4000 generations, cycling 10 sub-tasks
# (TASK_PERIOD=10), so each sub-task is seen twice. This sweep changes ONLY
# the switch interval. Everything else is held fixed:
#
#     total budget   4000 generations = 3.072e9 env steps (NE and RL alike)
#     sub-tasks      the same 10 per trial (the draw is seeded by the trial
#                    alone and nests, gymnax_classic._task_noise_vector_list)
#     cells          CartPole / Acrobot sigma 1.0, MountainCar sigma 0.1
#     arms           the reported ones: ga nes ppo trac redo cchain pbt pbt2
#
#     interval   phases   visits per sub-task   RL updates per phase
#       50         80            8                    375
#      200         20            2                   1500   <- runs_centroid
#      400         10            1                   3000
#
# Intervals must be even: a generation is 512 x 3 x 500 / 102400 = 7.5 PPO
# updates, and run_experiments.sh truncates the RL interval to an integer, so
# an odd one would put the PPO switches off the NE ones (CLAUDE.md rule c).
#
# 200 is NOT re-run: its trials 1..NUM_TRIALS are the same seeds (BASE_SEED
# 42) in projects/iclr_2027/runs_centroid.
#
# Each interval is its own tree, projects/iclr_2027/runs_freq/interval<N>,
# with the usual <suite>/continual/<arm>/<cell>/trial_<k> shape underneath.
# run_condition skips finished trials, so this is safe to restart.
#
# Env: INTERVALS ("50 400"), ARMS, NUM_TRIALS (5), GPUS (default: GPUs 0-2,
#      4 jobs each -- 3-7 are the cheetah diversity runs), DRY_RUN.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

INTERVALS="${INTERVALS:-50 400}"
ARMS="${ARMS:-ga nes ppo trac redo cchain pbt pbt2}"
NUM_TRIALS="${NUM_TRIALS:-5}"
TOTAL_GENS=4000
export GPUS="${GPUS:-0 0 0 0 1 1 1 1 2 2 2 2}"
export DRY_RUN="${DRY_RUN:-0}"

for iv in $INTERVALS; do
    if [ $((iv % 2)) -ne 0 ] || [ $((TOTAL_GENS % iv)) -ne 0 ]; then
        echo "FATAL: interval $iv must be even and divide $TOTAL_GENS" >&2
        exit 1
    fi
done

for iv in $INTERVALS; do
    ROOT="projects/iclr_2027/runs_freq/interval$iv"
    NT=$((TOTAL_GENS / iv))
    echo "=== $(date -Is) interval $iv: $NT phases -> $ROOT ==="

    NUM_TRIALS="$NUM_TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
      ENVS="CartPole-v1 Acrobot-v1" \
      NUM_TASKS="$NT" TASK_PERIOD=10 TASK_INTERVAL="$iv" SIGMAS=1.0 \
      LOG_DIR="logs/launch_freq_interval${iv}_sigma1.0" \
      bash scripts/train/launch.sh gymnax_continual $ARMS

    NUM_TRIALS="$NUM_TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
      ENVS="MountainCar-v0" \
      NUM_TASKS="$NT" TASK_PERIOD=10 TASK_INTERVAL="$iv" SIGMAS=0.1 \
      LOG_DIR="logs/launch_freq_interval${iv}_mcar_sigma0.1" \
      bash scripts/train/launch.sh gymnax_continual $ARMS
done

echo "=== $(date -Is) done ==="
echo "failures:"
grep -hv 'exit=0' logs/launch_freq_*/status.tsv 2>/dev/null || echo "  none"
