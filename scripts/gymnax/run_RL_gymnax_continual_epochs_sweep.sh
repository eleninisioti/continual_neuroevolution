#!/bin/bash
# Sweep num_epochs for PPO continual on Gymnax environments.
# Purpose: Address reviewer concern about optimization depth — isolate the effect
# of gradient update frequency on plasticity loss while keeping total env steps fixed.
#
# Usage:
#   ./run_RL_gymnax_continual_epochs_sweep.sh        # All envs on GPU 0
#   GPU=3 ./run_RL_gymnax_continual_epochs_sweep.sh  # All envs on GPU 3

set -e

GPU="${GPU:-3}"
NUM_TRIALS="${NUM_TRIALS:-5}"
ENVS="${ENVS:- MountainCar-v0}"
EPOCHS=(1 2 4 8 10)
METHOD="ppo"
WANDB_PROJECT="RL_epochs_sweep"

echo "=========================================="
echo "PPO Continual — num_epochs Sweep"
echo "=========================================="
echo "GPU: $GPU"
echo "Trials: $NUM_TRIALS"
echo "Environments: $ENVS"
echo "Epochs: ${EPOCHS[*]}"
echo "Method: $METHOD"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TRAIN_SCRIPT="$REPO_ROOT/source/gymnax/train_RL_gymnax_continual.py"

for ENV in $ENVS; do
    for NUM_EPOCH in "${EPOCHS[@]}"; do
        for TRIAL in $(seq 1 $NUM_TRIALS); do
            OUTPUT_DIR="projects/gymnax/rl_epochs_sweep/${ENV}/epochs_${NUM_EPOCH}/trial_${TRIAL}"
            echo ""
            echo ">>> $METHOD | $ENV | epochs=$NUM_EPOCH | Trial $TRIAL"

            CUDA_VISIBLE_DEVICES=$GPU python "$TRAIN_SCRIPT" \
                --env "$ENV" \
                --method "$METHOD" \
                --trial "$TRIAL" \
                --gpus "$GPU" \
                --num_epochs "$NUM_EPOCH" \
                --noise_range 1.0 \
                --output_dir "$OUTPUT_DIR" \
                --wandb_project "$WANDB_PROJECT"
        done
    done
done

echo ""
echo "=========================================="
echo "PPO num_epochs sweep complete!"
echo "Results in projects/gymnax/rl_epochs_sweep/"
echo "=========================================="
