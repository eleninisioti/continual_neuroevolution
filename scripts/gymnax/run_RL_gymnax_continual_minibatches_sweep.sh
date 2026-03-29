#!/bin/bash
# Sweep num_minibatches for PPO continual on Gymnax environments (num_epochs fixed at 8).
# Purpose: Vary gradient updates per task through minibatch count instead of epochs.
# This changes both the number of gradient updates AND the minibatch size:
#   grad_updates_per_update = num_epochs(8) × num_minibatches
#   minibatch_size = batch_size(102400) / num_minibatches
# Total env steps stay fixed across all runs (512M).
#
# Usage:
#   ./run_RL_gymnax_continual_minibatches_sweep.sh        # All envs on GPU 0
#   GPU=3 ./run_RL_gymnax_continual_minibatches_sweep.sh  # All envs on GPU 3

set -e

GPU="${GPU:-3}"
NUM_TRIALS="${NUM_TRIALS:-5}"
ENVS="${ENVS:-Acrobot-v1 CartPole-v1 MountainCar-v0}"
MINIBATCHES=(1 4 8 16 32 64)
NUM_EPOCHS=10
METHOD="ppo"
WANDB_PROJECT="RL_minibatches_sweep"

echo "=========================================="
echo "PPO Continual — num_minibatches Sweep"
echo "=========================================="
echo "GPU: $GPU"
echo "Trials: $NUM_TRIALS"
echo "Environments: $ENVS"
echo "num_epochs (fixed): $NUM_EPOCHS"
echo "num_minibatches: ${MINIBATCHES[*]}"
echo "Method: $METHOD"
echo ""
echo "Grad updates per PPO update = num_epochs × num_minibatches:"
for MB in "${MINIBATCHES[@]}"; do
    GRAD=$((NUM_EPOCHS * MB))
    MB_SIZE=$((102400 / MB))
    echo "  minibatches=$MB → $GRAD grad updates (minibatch_size=$MB_SIZE)"
done
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TRAIN_SCRIPT="$REPO_ROOT/source/gymnax/train_RL_gymnax_continual.py"

for ENV in $ENVS; do
    for MB in "${MINIBATCHES[@]}"; do
        for TRIAL in $(seq 1 $NUM_TRIALS); do
            OUTPUT_DIR="projects/gymnax/rl_minibatches_sweep/${ENV}/minibatches_${MB}/trial_${TRIAL}"
            echo ""
            echo ">>> $METHOD | $ENV | minibatches=$MB | Trial $TRIAL"

            CUDA_VISIBLE_DEVICES=$GPU python "$TRAIN_SCRIPT" \
                --env "$ENV" \
                --method "$METHOD" \
                --trial "$TRIAL" \
                --gpus "$GPU" \
                --num_epochs "$NUM_EPOCHS" \
                --num_minibatches "$MB" \
                --noise_range 1.0 \
                --output_dir "$OUTPUT_DIR" \
                --wandb_project "$WANDB_PROJECT"
        done
    done
done

echo ""
echo "=========================================="
echo "PPO num_minibatches sweep complete!"
echo "Results in projects/gymnax/rl_minibatches_sweep/"
echo "=========================================="
