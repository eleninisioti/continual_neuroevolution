#!/bin/bash
# Run DNS (Dominated Novelty Search) CheetahRun continual friction training - 10 trials
# Runs trials sequentially on a single GPU
#
# Usage: ./run_DNS_cheetah_continual.sh [GPU_ID]

set -e

GPU_ID=${1:-3}

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
ENV="CheetahRun"

# Task settings
NUM_TASKS=30
EPISODES_PER_TASK=200

# DNS settings
POP_SIZE=512
BATCH_SIZE=256
K=3
ISO_SIGMA=0.05
LINE_SIGMA=0.5

# Friction multipliers
FRICTION_DEFAULT=1.0
FRICTION_LOW=0.2
FRICTION_HIGH=5.0

# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================
mkdir -p "$BASE_DIR"

echo "=============================================="
echo "DNS CheetahRun Continual Friction Training"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Trials: $NUM_TRIALS"
echo ""
echo "Experiment: Friction changes between tasks"
echo "  - $NUM_TASKS tasks per trial"
echo "  - Friction cycles: default ($FRICTION_DEFAULT) -> low ($FRICTION_LOW) -> high ($FRICTION_HIGH)"
echo "  - $EPISODES_PER_TASK episodes per task"
echo "  - Population size: $POP_SIZE, Batch size: $BATCH_SIZE"
echo "  - k (novelty neighbors): $K"
echo ""

# Run 10 trials sequentially
for TRIAL in $(seq 1 $NUM_TRIALS); do
    SEED=$((42 + TRIAL - 1))
    
    RUN_NAME="dns_${ENV}_continual_friction_trial${TRIAL}"
    PROJECT_DIR="${BASE_DIR}/dns_${ENV}_continual_friction"
    OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "[GPU $GPU_ID] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."
    
    LOG_FILE="${OUTPUT_DIR}/train.log"
    python source/mujoco/train_DNS_cheetah_continual.py \
        --env $ENV \
        --task_mod friction \
        --friction_default_mult $FRICTION_DEFAULT \
        --friction_low_mult $FRICTION_LOW \
        --friction_high_mult $FRICTION_HIGH \
        --num_tasks $NUM_TASKS \
        --episodes_per_task $EPISODES_PER_TASK \
        --pop_size $POP_SIZE \
        --batch_size $BATCH_SIZE \
        --k $K \
        --iso_sigma $ISO_SIGMA \
        --line_sigma $LINE_SIGMA \
        --seed $SEED \
        --gpus $GPU_ID \
        --run_name $RUN_NAME \
        --output_dir $OUTPUT_DIR \
        --wandb_project ${WANDB_PROJECT}_dns \
        > "$LOG_FILE" 2>&1
    
    echo "[GPU $GPU_ID] Completed: $RUN_NAME"
done

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=============================================="
echo "  DNS: $NUM_TRIALS trials"
echo "  Output: $BASE_DIR"
echo "=============================================="
