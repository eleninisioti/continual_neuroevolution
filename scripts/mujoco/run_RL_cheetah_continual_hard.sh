#!/bin/bash
# Run all CheetahRun continual friction experiments (HARD settings matching PBT paper)
# 
# Paper settings:
#   - Friction log-uniform from [0.02, 2.00]
#   - Change every 2M timesteps
#
# Methods: PPO, ReDo-PPO, TRAC-PPO
# Total runs: 3 methods × 10 trials = 30 runs
#
# Usage: ./run_RL_cheetah_continual_hard.sh

set -e

# GPU assignments for parallel execution
GPU_PPO=0
GPU_TRAC=2
GPU_REDO=4

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
# CONFIGURATION (HARD - matching PBT paper)
# ============================================
WANDB_PROJECT="continual_neuroevolution_ICML_2026_hard"
BASE_DIR="projects/mujoco"
NUM_TRIALS=10

# Environment
ENV="CheetahRun"

# PPO settings - HARD: 2M timesteps per task (paper setting)
PPO_STEPS_PER_TASK=2000000

# Task settings - more tasks since each is shorter
NUM_TASKS=100  # 100 tasks × 2M = 200M total (similar total budget)

# Friction multipliers - HARD: paper range [0.02, 2.0]
FRICTION_LOW=0.02
FRICTION_HIGH=2.0

# Sampling mode - log-uniform as in paper
SAMPLING_MODE="log_uniform"

# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================
mkdir -p "$BASE_DIR"

echo "=============================================="
echo "CheetahRun Continual Friction - HARD (Paper Settings)"
echo "=============================================="
echo "GPUs: PPO=$GPU_PPO, TRAC=$GPU_TRAC, ReDo=$GPU_REDO"
echo "Methods: PPO, ReDo-PPO, TRAC-PPO"
echo "Trials per method: $NUM_TRIALS"
echo ""
echo "HARD settings (matching PBT paper):"
echo "  - Friction range: [$FRICTION_LOW, $FRICTION_HIGH] (log-uniform)"
echo "  - Timesteps per task: $PPO_STEPS_PER_TASK (2M)"
echo "  - Number of tasks: $NUM_TASKS"
echo "=============================================="

# ============================================
# Function to run PPO trials
# ============================================
run_ppo() {
    local GPU_ID=$GPU_PPO
    echo ""
    echo "=============================================="
    echo "PPO (baseline) on GPU $GPU_ID - HARD"
    echo "=============================================="

    for TRIAL in $(seq 1 $NUM_TRIALS); do
        SEED=$((42 + TRIAL - 1))
        
        RUN_NAME="ppo_${ENV}_continual_friction_hard_trial${TRIAL}"
        PROJECT_DIR="${BASE_DIR}/ppo_${ENV}_continual_friction_hard"
        OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
        mkdir -p "$OUTPUT_DIR"
        
        echo "[GPU $GPU_ID] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."
        
        LOG_FILE="${OUTPUT_DIR}/train.log"
        python source/mujoco/train_RL_cheetah_continual.py \
            --env $ENV \
            --task_mod friction \
            --friction_low_mult $FRICTION_LOW \
            --friction_high_mult $FRICTION_HIGH \
            --sampling_mode $SAMPLING_MODE \
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
    local GPU_ID=$GPU_REDO
    echo ""
    echo "=============================================="
    echo "ReDo-PPO on GPU $GPU_ID - HARD"
    echo "=============================================="

    for TRIAL in $(seq 1 $NUM_TRIALS); do
        SEED=$((42 + TRIAL - 1))
        
        RUN_NAME="redo_ppo_${ENV}_continual_friction_hard_trial${TRIAL}"
        PROJECT_DIR="${BASE_DIR}/redo_ppo_${ENV}_continual_friction_hard"
        OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
        mkdir -p "$OUTPUT_DIR"
        
        echo "[GPU $GPU_ID] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."
        
        LOG_FILE="${OUTPUT_DIR}/train.log"
        python source/mujoco/train_RL_cheetah_continual.py \
            --env $ENV \
            --task_mod friction \
            --friction_low_mult $FRICTION_LOW \
            --friction_high_mult $FRICTION_HIGH \
            --sampling_mode $SAMPLING_MODE \
            --num_tasks $NUM_TASKS \
            --timesteps_per_task $PPO_STEPS_PER_TASK \
            --seed $SEED \
            --gpus $GPU_ID \
            --run_name $RUN_NAME \
            --output_dir $OUTPUT_DIR \
            --wandb_project ${WANDB_PROJECT}_redo \
            --use_redo \
            --redo_frequency 10 \
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
    local GPU_ID=$GPU_TRAC
    echo ""
    echo "=============================================="
    echo "TRAC-PPO on GPU $GPU_ID - HARD"
    echo "=============================================="

    for TRIAL in $(seq 1 $NUM_TRIALS); do
        SEED=$((42 + TRIAL - 1))
        
        RUN_NAME="trac_ppo_${ENV}_continual_friction_hard_trial${TRIAL}"
        PROJECT_DIR="${BASE_DIR}/trac_ppo_${ENV}_continual_friction_hard"
        OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
        mkdir -p "$OUTPUT_DIR"
        
        echo "[GPU $GPU_ID] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."
        
        LOG_FILE="${OUTPUT_DIR}/train.log"
        python source/mujoco/train_RL_cheetah_continual.py \
            --env $ENV \
            --task_mod friction \
            --friction_low_mult $FRICTION_LOW \
            --friction_high_mult $FRICTION_HIGH \
            --sampling_mode $SAMPLING_MODE \
            --num_tasks $NUM_TASKS \
            --timesteps_per_task $PPO_STEPS_PER_TASK \
            --seed $SEED \
            --gpus $GPU_ID \
            --run_name $RUN_NAME \
            --output_dir $OUTPUT_DIR \
            --wandb_project ${WANDB_PROJECT}_trac \
            --trac \
            > "$LOG_FILE" 2>&1
        
        echo "[GPU $GPU_ID] Completed: $RUN_NAME"
    done

    echo ""
    echo "TRAC-PPO experiments complete!"
}

# ============================================
# Run all methods in parallel
# ============================================
echo ""
echo "Starting all methods in parallel..."
echo "  PPO on GPU $GPU_PPO"
echo "  ReDo-PPO on GPU $GPU_REDO"
echo "  TRAC-PPO on GPU $GPU_TRAC"
echo ""

run_ppo &
PID_PPO=$!

run_redo &
PID_REDO=$!

run_trac &
PID_TRAC=$!

# Wait for all to complete
echo "Waiting for all experiments to complete..."
wait $PID_PPO
echo "PPO finished!"
wait $PID_REDO
echo "ReDo-PPO finished!"
wait $PID_TRAC
echo "TRAC-PPO finished!"

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=============================================="
echo "Results saved to:"
echo "  - ${BASE_DIR}/ppo_${ENV}_continual_friction_hard/"
echo "  - ${BASE_DIR}/redo_ppo_${ENV}_continual_friction_hard/"
echo "  - ${BASE_DIR}/trac_ppo_${ENV}_continual_friction_hard/"
echo "=============================================="
