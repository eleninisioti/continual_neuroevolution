#!/bin/bash
# Run PBT + PPO Continual on Gymnax environments with population size sweep
#
# Usage:
#   PBT_MODE=weights_only ./run_PBT_gymnax_continual_pop_sweep.sh
#   PBT_MODE=full GPU=1 ENVS="CartPole-v1" ./run_PBT_gymnax_continual_pop_sweep.sh

set -e

# Configuration
GPU="${GPU:-3}"
NUM_TRIALS="${NUM_TRIALS:-5}"
ENVS="${ENVS:-CartPole-v1 Acrobot-v1 MountainCar-v0}"
POP_SIZES="${POP_SIZES:-2 4 8 16 32 64 128 256 512}"
PBT_INTERVAL="${PBT_INTERVAL:-10}"
PBT_MODE="${PBT_MODE:-weights_only}"  # "full" or "weights_only"
WANDB_PROJECT="PBT_popsize_study"

echo "=========================================="
echo "PBT + PPO Continual Population Sweep"
echo "=========================================="
echo "GPU: $GPU"
echo "Trials: $NUM_TRIALS"
echo "Environments: $ENVS"
echo "Population sizes: $POP_SIZES"
echo "PBT interval: $PBT_INTERVAL"
echo "PBT mode: $PBT_MODE"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

for ENV in $ENVS; do
    echo ""
    echo "=========================================="
    echo "PBT Continual Population Sweep on $ENV"
    echo "=========================================="

    for POP_SIZE in $POP_SIZES; do
        echo ""
        echo "=========================================="
        echo "$ENV - pop_size=$POP_SIZE"
        echo "=========================================="

        for TRIAL in $(seq 1 $NUM_TRIALS); do
            echo ""
            echo "Starting Trial $TRIAL / $NUM_TRIALS ($ENV, pop_size=$POP_SIZE)"
            echo "----------------------------------------"

            CUDA_VISIBLE_DEVICES=$GPU python "$REPO_ROOT/source/gymnax/train_PBT_gymnax_continual.py" \
                --env "$ENV" \
                --trial "$TRIAL" \
                --gpus "$GPU" \
                --pop_size "$POP_SIZE" \
                --pbt_interval "$PBT_INTERVAL" \
                --pbt_mode "$PBT_MODE" \
                --wandb_project "$WANDB_PROJECT"

            echo "Trial $TRIAL complete!"
        done

        echo ""
        echo "$ENV pop_size=$POP_SIZE: All $NUM_TRIALS trials complete!"
    done

    echo ""
    echo "=========================================="
    echo "PBT Continual on $ENV: All population sizes complete!"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All PBT Continual population sweep complete!"
echo "=========================================="
