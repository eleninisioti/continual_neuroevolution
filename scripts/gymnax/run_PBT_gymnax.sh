#!/bin/bash
# Run PBT + PPO on Gymnax environments (Non-Continual and Continual)
#
# Usage:
#   ./run_PBT_gymnax.sh                    # Run both modes on all envs
#   ./run_PBT_gymnax.sh continual          # Run only continual mode
#   ./run_PBT_gymnax.sh noncontinual       # Run only non-continual mode

set -e

# Configuration
GPU="${GPU:-3}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MODE="${1:-noncontinual continual}"
ENVS="${ENVS:-MountainCar-v0}"
POP_SIZE="${POP_SIZE:-512}"
PBT_INTERVAL="${PBT_INTERVAL:-1}"
PBT_MODE="${PBT_MODE:-full}"

echo "=========================================="
echo "PBT + PPO on Gymnax"
echo "=========================================="
echo "GPU: $GPU"
echo "Trials: $NUM_TRIALS"
echo "Mode(s): $MODE"
echo "Environments: $ENVS"
echo "Population size: $POP_SIZE"
echo "PBT interval: $PBT_INTERVAL"
echo "PBT mode: $PBT_MODE"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

for ENV in $ENVS; do
    for M in $MODE; do
        if [ "$M" = "noncontinual" ]; then
            TRAIN_SCRIPT="$REPO_ROOT/source/gymnax/train_PBT_gymnax.py"
            LABEL="Non-Continual"
        elif [ "$M" = "continual" ]; then
            TRAIN_SCRIPT="$REPO_ROOT/source/gymnax/train_PBT_gymnax_continual.py"
            LABEL="Continual"
        else
            echo "Unknown mode: $M (use 'noncontinual' or 'continual')"
            continue
        fi

        echo ""
        echo "=========================================="
        echo "PBT on $ENV ($LABEL)"
        echo "=========================================="

        for TRIAL in $(seq 1 $NUM_TRIALS); do
            echo ""
            echo "Starting Trial $TRIAL / $NUM_TRIALS (PBT $LABEL on $ENV)"
            echo "----------------------------------------"

            CUDA_VISIBLE_DEVICES=$GPU python "$TRAIN_SCRIPT" \
                --env "$ENV" \
                --trial "$TRIAL" \
                --gpus "$GPU" \
                --pop_size "$POP_SIZE" \
                --pbt_interval "$PBT_INTERVAL" \
                --pbt_mode "$PBT_MODE"

            echo "Trial $TRIAL complete!"
        done

        echo ""
        echo "PBT $LABEL on $ENV: All $NUM_TRIALS trials complete!"
    done
done

echo ""
echo "=========================================="
echo "All PBT training complete!"
echo "=========================================="
