#!/bin/bash
# Run SimpleGA Go1 leg damage continual training - 10 trials across GPUs
# Runs one trial per GPU at a time, waits for completion before starting next

set -e

GPU_ID=${1:-5}

# ============================================
# ACTIVATE VIRTUAL ENVIRONMENT
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
if [ -f "${REPO_DIR}/.venv/bin/activate" ]; then
    source "${REPO_DIR}/.venv/bin/activate"
    echo "Activated virtual environment: ${REPO_DIR}/.venv"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Using active virtual environment: $VIRTUAL_ENV"
else
    echo "WARNING: No virtual environment found. Python commands may fail."
fi

cd "$REPO_DIR"

# ============================================
# CONFIGURATION
# ============================================
WANDB_PROJECT="continual_neuroevolution_ICML_2026"
BASE_DIR="projects/mujoco"
NUM_TRIALS=10

# Environment
ENV="Go1JoystickFlatTerrain"

# Task settings
NUM_TASKS=20
GENS_PER_TASK=50

# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================
mkdir -p "$BASE_DIR"

echo "=============================================="
echo "SimpleGA Go1 Leg Damage Continual Training"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Trials: $NUM_TRIALS"
echo ""
echo "Experiment: Each task damages a different leg"
echo "  - $NUM_TASKS tasks per trial (first task = healthy/no damage)"
echo "  - Random leg selection (avoiding consecutive same leg)"
echo "  - Damaged leg is LOCKED in bent position (frozen joints)"
echo "  - $GENS_PER_TASK generations per task"
echo ""

# Run 10 trials sequentially
for TRIAL in $(seq 1 $NUM_TRIALS); do
    SEED=$((42 + TRIAL - 1))
    
    RUN_NAME="ga_${ENV}_continual_legdamage_trial${TRIAL}"
    PROJECT_DIR="${BASE_DIR}/ga_${ENV}_continual_legdamage"
    OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "[GPU $GPU_ID] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."
    
    LOG_FILE="${OUTPUT_DIR}/train.log"
    PYTHONUNBUFFERED=1 python -u source/mujoco/train_GA_quadruped_continual.py \
        --env $ENV \
        --num_tasks $NUM_TASKS \
        --gens_per_task $GENS_PER_TASK \
        --seed $SEED \
        --trial $TRIAL \
        --gpus $GPU_ID \
        --pop_size 512 \
        --num_evals 3 \
        --run_name $RUN_NAME \
        --output_dir $OUTPUT_DIR \
        --wandb_project ${WANDB_PROJECT}_ga \
        > "$LOG_FILE" 2>&1
    
    echo "[GPU $GPU_ID] Completed: $RUN_NAME"
done

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=============================================="
echo "  SimpleGA: $NUM_TRIALS trials"
echo "  Output: $BASE_DIR"
echo "=============================================="
