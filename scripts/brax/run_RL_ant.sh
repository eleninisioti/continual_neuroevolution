#!/bin/bash
# Train PPO on Brax Ant (non-continual).
#
# Usage:
#   ./scripts/brax/run_RL_ant.sh          # default: 5 trials on GPU 0
#   GPU=1 NUM_TRIALS=3 ./scripts/brax/run_RL_ant.sh

set -e

GPU="${GPU:-0}"
NUM_TRIALS="${NUM_TRIALS:-5}"
METHODS="${METHODS:-ppo}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Activate venv
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
    echo "Activated virtual environment: ${REPO_ROOT}/.venv"
fi

ENV="ant"
WANDB_PROJECT="brax_ant"
BASE_DIR="projects/brax"

echo "=========================================="
echo "Brax Ant - PPO (Non-Continual)"
echo "=========================================="
echo "Trials: $NUM_TRIALS"
echo "Methods: $METHODS"
echo "GPU: $GPU"
echo "=========================================="

for method in $METHODS; do
    # Set method flags
    EXTRA_FLAGS=""
    if [ "$method" = "trac" ]; then
        EXTRA_FLAGS="--use_trac"
        ALGO_NAME="trac_ppo"
    elif [ "$method" = "redo" ]; then
        EXTRA_FLAGS="--use_redo --redo_frequency 10"
        ALGO_NAME="redo_ppo"
    else
        ALGO_NAME="ppo"
    fi

    for trial in $(seq 1 $NUM_TRIALS); do
        SEED=$((42 + trial))
        RUN_NAME="${ALGO_NAME}_${ENV}_trial${trial}"
        OUTPUT_DIR="${BASE_DIR}/${ALGO_NAME}_${ENV}/trial_${trial}"
        mkdir -p "$OUTPUT_DIR"

        echo ""
        echo ">>> $ENV | $ALGO_NAME | Trial $trial / $NUM_TRIALS -> $OUTPUT_DIR"

        CUDA_VISIBLE_DEVICES=$GPU python "$REPO_ROOT/source/brax/train_RL_ant.py" \
            --env "$ENV" \
            --seed "$SEED" \
            --trial "$trial" \
            --gpus "$GPU" \
            --output_dir "$OUTPUT_DIR" \
            --run_name "$RUN_NAME" \
            --wandb_project "${WANDB_PROJECT}_${ALGO_NAME}" \
            $EXTRA_FLAGS \
            2>&1 | tee "${OUTPUT_DIR}/train.log"

        echo ">>> Completed: $RUN_NAME"
    done
done

echo ""
echo "All runs complete!"
