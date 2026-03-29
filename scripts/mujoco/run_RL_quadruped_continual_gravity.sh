#!/bin/bash
# Script to run all Go1 quadruped continual GRAVITY experiments in parallel
# Methods: PPO, ReDo-PPO, TRAC-PPO (running in parallel on different GPUs)
# Total runs: 3 methods × 10 trials = 30 runs
#
# Gravity multipliers cycle: 1.0x (normal) → 0.2x (low) → 5.0x (high)
#
# Usage: ./run_RL_quadruped_continual_gravity.sh

set -e

# GPU assignment (all methods run sequentially on same GPU)
GPU="${GPU:-1}"

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
NUM_TRIALS="${NUM_TRIALS:-10}"

# Environment
ENV="Go1JoystickFlatTerrain"

# Steps per task (same as leg damage: 50 generations × 512,000 steps)
PPO_STEPS_PER_TASK="${PPO_STEPS_PER_TASK:-25600000}"

# Task settings
NUM_TASKS="${NUM_TASKS:-20}"

# Gravity multipliers (cycle: default → low → high)
GRAVITY_DEFAULT=1.0
GRAVITY_LOW=0.2
GRAVITY_HIGH=5.0

# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================
mkdir -p "$BASE_DIR"

echo "=============================================="
echo "Go1 Continual GRAVITY - ALL METHODS (SEQUENTIAL)"
echo "==============================================" 
echo "GPU: $GPU"
echo "Methods: PPO, ReDo-PPO, TRAC-PPO"
echo "Trials per method: $NUM_TRIALS"
echo "Total runs: $((NUM_TRIALS * 3))"
echo ""
echo "Experiment: Each task changes gravity"
echo "  - $NUM_TASKS tasks per trial"
echo "  - Gravity cycle: ${GRAVITY_DEFAULT}x → ${GRAVITY_LOW}x → ${GRAVITY_HIGH}x"
echo "  - $PPO_STEPS_PER_TASK steps per task"
echo ""

# ============================================
# Function to run PPO trials
# ============================================
run_ppo() {
    local GPU_ID=$GPU
    echo ""
    echo "=============================================="
    echo "PPO (baseline) on GPU $GPU_ID"
    echo "=============================================="

    for TRIAL in $(seq 1 $NUM_TRIALS); do
        SEED=$((42 + TRIAL))

        RUN_NAME="ppo_${ENV}_continual_gravity_trial${TRIAL}"
        PROJECT_DIR="${BASE_DIR}/ppo_${ENV}_continual_gravity"
        OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
        mkdir -p "$OUTPUT_DIR"

        echo "[GPU $GPU_ID] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."

        LOG_FILE="${OUTPUT_DIR}/train.log"
        python source/mujoco/train_RL_cheetah_continual.py \
            --env $ENV \
            --task_mod gravity \
            --gravity_default_mult $GRAVITY_DEFAULT \
            --gravity_low_mult $GRAVITY_LOW \
            --gravity_high_mult $GRAVITY_HIGH \
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
}

# ============================================
# Function to run ReDo-PPO trials
# ============================================
run_redo() {
    local GPU_ID=$GPU
    echo ""
    echo "=============================================="
    echo "ReDo-PPO on GPU $GPU_ID"
    echo "=============================================="

    for TRIAL in $(seq 1 $NUM_TRIALS); do
        SEED=$((42 + TRIAL))

        RUN_NAME="redo_ppo_${ENV}_continual_gravity_trial${TRIAL}"
        PROJECT_DIR="${BASE_DIR}/redo_ppo_${ENV}_continual_gravity"
        OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
        mkdir -p "$OUTPUT_DIR"

        echo "[GPU $GPU_ID] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."

        LOG_FILE="${OUTPUT_DIR}/train.log"
        python source/mujoco/train_RL_cheetah_continual.py \
            --env $ENV \
            --task_mod gravity \
            --gravity_default_mult $GRAVITY_DEFAULT \
            --gravity_low_mult $GRAVITY_LOW \
            --gravity_high_mult $GRAVITY_HIGH \
            --num_tasks $NUM_TASKS \
            --timesteps_per_task $PPO_STEPS_PER_TASK \
            --seed $SEED \
            --gpus $GPU_ID \
            --run_name $RUN_NAME \
            --output_dir $OUTPUT_DIR \
            --use_redo \
            --redo_frequency 1 \
            --wandb_project ${WANDB_PROJECT}_redo \
            > "$LOG_FILE" 2>&1

        echo "[GPU $GPU_ID] Completed: $RUN_NAME"
    done

    echo ""
    echo "ReDo-PPO experiments complete!"
}

# ============================================
# Function to run TRAC-PPO trials
# ============================================
run_trac() {
    local GPU_ID=$GPU
    echo ""
    echo "=============================================="
    echo "TRAC-PPO on GPU $GPU_ID"
    echo "=============================================="

    for TRIAL in $(seq 1 $NUM_TRIALS); do
        SEED=$((42 + TRIAL))

        RUN_NAME="trac_ppo_${ENV}_continual_gravity_trial${TRIAL}"
        PROJECT_DIR="${BASE_DIR}/trac_ppo_${ENV}_continual_gravity"
        OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
        mkdir -p "$OUTPUT_DIR"

        echo "[GPU $GPU_ID] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."

        LOG_FILE="${OUTPUT_DIR}/train.log"
        python source/mujoco/train_RL_cheetah_continual.py \
            --env $ENV \
            --task_mod gravity \
            --gravity_default_mult $GRAVITY_DEFAULT \
            --gravity_low_mult $GRAVITY_LOW \
            --gravity_high_mult $GRAVITY_HIGH \
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
}

# ============================================
# Run all methods sequentially on same GPU
# ============================================
echo "Running all methods sequentially on GPU $GPU..."
run_ppo
run_redo
run_trac

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=============================================="
echo "  PPO: $NUM_TRIALS trials"
echo "  TRAC-PPO: $NUM_TRIALS trials"
echo "  ReDo-PPO: $NUM_TRIALS trials"
echo "  GPU: $GPU"
echo "  Total: $((NUM_TRIALS * 3)) runs"
echo "  Output: $BASE_DIR"
echo "=============================================="
