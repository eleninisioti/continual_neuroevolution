#!/bin/bash
# Train PPO (continual) on Brax Ant with gravity changes.
# Samples gravity randomly (log-uniform) between LOW and HIGH each task (Dohare et al. setup)
#
# Usage:
#   ./scripts/brax/run_RL_ant_continual_gravity.sh
#   GPU=2 NUM_TRIALS=5 ./scripts/brax/run_RL_ant_continual_gravity.sh

set -e

GPU="${GPU:-2}"
NUM_TRIALS="${NUM_TRIALS:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Activate venv
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
    echo "Activated virtual environment: ${REPO_ROOT}/.venv"
fi

WANDB_PROJECT="brax_ant_continual"
BASE_DIR="projects/brax"

# Continual learning settings (HARD - matching PBT paper frequency)
NUM_TASKS=30
TIMESTEPS_PER_TASK=6000000

# Gravity multipliers (random log-uniform sampling between LOW and HIGH)
GRAVITY_LOW=0.6
GRAVITY_HIGH=3.0

echo "=========================================="
echo "Brax Ant - PPO Continual (Gravity)"
echo "=========================================="
echo "Trials: $NUM_TRIALS"
echo "Methods: ppo, redo, trac"
echo "GPU: $GPU"
echo "Tasks: $NUM_TASKS"
echo "Timesteps/task: $TIMESTEPS_PER_TASK"
echo "Gravity: [${GRAVITY_LOW}x, ${GRAVITY_HIGH}x] (random log-uniform)"
echo "=========================================="

run_method() {
    local method=$1
    local gpu=$2
    
    EXTRA_FLAGS=""
    if [ "$method" = "trac" ]; then
        EXTRA_FLAGS="--use_trac"
        ALGO_NAME="trac_ppo"
    elif [ "$method" = "redo" ]; then
        EXTRA_FLAGS="--use_redo --redo_frequency 1"
        ALGO_NAME="redo_ppo"
    else
        ALGO_NAME="ppo"
    fi

    for trial in $(seq 1 $NUM_TRIALS); do
        SEED=$((42 + trial - 1))
        RUN_NAME="${ALGO_NAME}_ant_random_gravity_trial${trial}"
        OUTPUT_DIR="${BASE_DIR}/${ALGO_NAME}_ant_random_gravity/trial_${trial}"
        mkdir -p "$OUTPUT_DIR"

        echo ""
        echo ">>> ant | $ALGO_NAME | GPU $gpu | Trial $trial / $NUM_TRIALS -> $OUTPUT_DIR"

        CUDA_VISIBLE_DEVICES=$gpu python "$REPO_ROOT/source/brax/train_RL_ant_continual.py" \
            --num_tasks $NUM_TASKS \
            --timesteps_per_task $TIMESTEPS_PER_TASK \
            --gravity_low_mult $GRAVITY_LOW \
            --gravity_high_mult $GRAVITY_HIGH \
            --seed "$SEED" \
            --trial "$trial" \
            --gpus "$gpu" \
            --output_dir "$OUTPUT_DIR" \
            --run_name "$RUN_NAME" \
            --wandb_project "${WANDB_PROJECT}" \
            --track_dormant \
            $EXTRA_FLAGS \
            2>&1 | tee "${OUTPUT_DIR}/train.log"

        echo ">>> Completed: $RUN_NAME"
    done
}

# Run methods sequentially on same GPU
echo "Starting methods sequentially on GPU $GPU..."

run_method "ppo" "$GPU"
echo "PPO finished!"

run_method "redo" "$GPU"
echo "ReDo finished!"

run_method "trac" "$GPU"
echo "TRAC finished!"

echo ""
echo "All runs complete!"
