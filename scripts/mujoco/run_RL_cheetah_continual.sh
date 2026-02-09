#!/bin/bash
# Script to run all CheetahRun continual friction experiments sequentially
# Methods: PPO, ReDo-PPO, TRAC-PPO (in that order)
# Total runs: 3 methods × 10 trials = 30 runs
#
# Usage: ./run_cheetah_continual_friction_all.sh [GPU_ID]

set -e

GPU_ID=${1:-0}

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

# PPO settings
PPO_STEPS_PER_TASK=51200000

# Task settings
NUM_TASKS=30

# Friction multipliers
FRICTION_DEFAULT=1.0
FRICTION_LOW=0.2
FRICTION_HIGH=5.0

# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================
mkdir -p "$BASE_DIR"

echo "=============================================="
echo "CheetahRun Continual Friction - ALL METHODS"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Methods: PPO, ReDo-PPO, TRAC-PPO"
echo "Trials per method: $NUM_TRIALS"
echo "Total runs: $((NUM_TRIALS * 3))"
echo ""

# ============================================
# 1. PPO (baseline)
# ============================================
echo ""
echo "=============================================="
echo "PHASE 1: PPO (baseline)"
echo "=============================================="

for TRIAL in $(seq 1 $NUM_TRIALS); do
    SEED=$((42 + TRIAL - 1))
    
    RUN_NAME="ppo_${ENV}_continual_friction_trial${TRIAL}"
    PROJECT_DIR="${BASE_DIR}/ppo_${ENV}_continual_friction"
    OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "[GPU $GPU_ID] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."
    
    LOG_FILE="${OUTPUT_DIR}/train.log"
    python source/mujoco/train_RL_cheetah_continual.py \
        --env $ENV \
        --task_mod friction \
        --friction_default_mult $FRICTION_DEFAULT \
        --friction_low_mult $FRICTION_LOW \
        --friction_high_mult $FRICTION_HIGH \
        --num_tasks $NUM_TASKS \
        --timesteps_per_task $PPO_STEPS_PER_TASK \
        --seed $SEED \
        --gpus $GPU_ID \
        --run_name $RUN_NAME \
        --output_dir $OUTPUT_DIR \
        --wandb_project ${WANDB_PROJECT}_ppo \
        > "$LOG_FILE" 2>&1
    
    echo "[GPU $GPU_ID] Completed: $RUN_NAME"
done

echo ""
echo "PPO experiments complete!"

# ============================================
# 2. ReDo-PPO
# ============================================
echo ""
echo "=============================================="
echo "PHASE 2: ReDo-PPO"
echo "=============================================="

for TRIAL in $(seq 1 $NUM_TRIALS); do
    SEED=$((42 + TRIAL - 1))
    
    RUN_NAME="redo_ppo_${ENV}_continual_friction_trial${TRIAL}"
    PROJECT_DIR="${BASE_DIR}/redo_ppo_${ENV}_continual_friction"
    OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "[GPU $GPU_ID] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."
    
    LOG_FILE="${OUTPUT_DIR}/train.log"
    python source/mujoco/train_RL_cheetah_continual.py \
        --env $ENV \
        --task_mod friction \
        --friction_default_mult $FRICTION_DEFAULT \
        --friction_low_mult $FRICTION_LOW \
        --friction_high_mult $FRICTION_HIGH \
        --num_tasks $NUM_TASKS \
        --timesteps_per_task $PPO_STEPS_PER_TASK \
        --seed $SEED \
        --gpus $GPU_ID \
        --run_name $RUN_NAME \
        --output_dir $OUTPUT_DIR \
        --redo \
        --redo_frequency 1 \
        --wandb_project ${WANDB_PROJECT}_redo \
        > "$LOG_FILE" 2>&1
    
    echo "[GPU $GPU_ID] Completed: $RUN_NAME"
done

echo ""
echo "ReDo-PPO experiments complete!"

# ============================================
# 3. TRAC-PPO
# ============================================
echo ""
echo "=============================================="
echo "PHASE 3: TRAC-PPO"
echo "=============================================="

for TRIAL in $(seq 1 $NUM_TRIALS); do
    SEED=$((42 + TRIAL - 1))
    
    RUN_NAME="trac_ppo_${ENV}_continual_friction_trial${TRIAL}"
    PROJECT_DIR="${BASE_DIR}/trac_ppo_${ENV}_continual_friction"
    OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "[GPU $GPU_ID] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."
    
    LOG_FILE="${OUTPUT_DIR}/train.log"
    python source/mujoco/train_RL_cheetah_continual.py \
        --env $ENV \
        --task_mod friction \
        --friction_default_mult $FRICTION_DEFAULT \
        --friction_low_mult $FRICTION_LOW \
        --friction_high_mult $FRICTION_HIGH \
        --num_tasks $NUM_TASKS \
        --timesteps_per_task $PPO_STEPS_PER_TASK \
        --seed $SEED \
        --gpus $GPU_ID \
        --run_name $RUN_NAME \
        --output_dir $OUTPUT_DIR \
        --trac \
        --wandb_project ${WANDB_PROJECT}_trac \
        > "$LOG_FILE" 2>&1
    
    echo "[GPU $GPU_ID] Completed: $RUN_NAME"
done

echo ""
echo "TRAC-PPO experiments complete!"

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=============================================="
echo "  PPO: $NUM_TRIALS trials"
echo "  ReDo-PPO: $NUM_TRIALS trials"
echo "  TRAC-PPO: $NUM_TRIALS trials"
echo "  Total: $((NUM_TRIALS * 3)) runs"
echo "  Output: $BASE_DIR"
echo "=============================================="
