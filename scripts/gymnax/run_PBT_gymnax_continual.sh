#!/bin/bash
# Run PBT + PPO Continual on Gymnax environments
#
# Usage:
#   PBT_MODE=full ./run_PBT_gymnax_continual.sh
#   PBT_MODE=weights_only GPU=1 ./run_PBT_gymnax_continual.sh

set -e

# Configuration
GPU="${GPU:-3}"
NUM_TRIALS="${NUM_TRIALS:-10}"
ENVS="${ENVS:-CartPole-v1 Acrobot-v1 MountainCar-v0}"
PBT_INTERVAL="${PBT_INTERVAL:-10}"
PBT_MODE="${PBT_MODE:-weights_only}"  # "full" or "weights_only"

# Per-environment population sizes
declare -A POP_SIZES
POP_SIZES["CartPole-v1"]=8
POP_SIZES["Acrobot-v1"]=8
POP_SIZES["MountainCar-v0"]=128

echo "=========================================="
echo "PBT + PPO Continual on Gymnax"
echo "=========================================="
echo "GPU: $GPU"
echo "Trials: $NUM_TRIALS"
echo "Environments: $ENVS"
echo "Pop sizes: CartPole=8, Acrobot=8, MountainCar=128"
echo "PBT interval: $PBT_INTERVAL"
echo "PBT mode: $PBT_MODE"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

for ENV in $ENVS; do
    POP_SIZE=${POP_SIZES[$ENV]:-8}
    
    echo ""
    echo "=========================================="
    echo "PBT Continual on $ENV (pop_size=$POP_SIZE)"
    echo "=========================================="

    for TRIAL in $(seq 1 $NUM_TRIALS); do
        echo ""
        echo "Starting Trial $TRIAL / $NUM_TRIALS (PBT Continual on $ENV)"
        echo "----------------------------------------"

        CUDA_VISIBLE_DEVICES=$GPU python "$REPO_ROOT/source/gymnax/train_PBT_gymnax_continual.py" \
            --env "$ENV" \
            --trial "$TRIAL" \
            --gpus "$GPU" \
            --pop_size "$POP_SIZE" \
            --pbt_interval "$PBT_INTERVAL" \
            --pbt_mode "$PBT_MODE"

        echo "Trial $TRIAL complete!"
    done

    echo ""
    echo "PBT Continual on $ENV: All $NUM_TRIALS trials complete!"
done

echo ""
echo "=========================================="
echo "All PBT Continual training complete!"
echo "=========================================="
