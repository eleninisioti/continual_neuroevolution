#!/bin/bash
# Script to run all Go1 leg damage continual experiments in parallel
# Methods: PPO (GPU 0), ReDo-PPO (GPU 1), TRAC-PPO (GPU 2)
# Total runs: 3 methods × 10 trials = 30 runs
#
# Usage: ./run_ppo_go1_legdamage_10trials.sh

set -e

# GPU assignments for parallel execution
export GPU_PPO=0
export GPU_REDO=1
export GPU_TRAC=2

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
export WANDB_PROJECT="continual_neuroevolution_ICML_2026"
export BASE_DIR="projects/mujoco"
export NUM_TRIALS=10

# Environment
export ENV="Go1JoystickFlatTerrain"

# PPO settings
export PPO_STEPS_PER_TASK=25600000

# Task settings
export NUM_TASKS=20

# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================
mkdir -p "$BASE_DIR"

echo "=============================================="
echo "Go1 Leg Damage Continual - ALL METHODS (PARALLEL)"
echo "=============================================="
echo "GPUs: PPO=$GPU_PPO, ReDo-PPO=$GPU_REDO, TRAC-PPO=$GPU_TRAC"
echo "Methods: PPO, ReDo-PPO, TRAC-PPO"
echo "Trials per method: $NUM_TRIALS"
echo "Total runs: $((NUM_TRIALS * 3))"
echo ""
echo "Experiment: Each task damages a different leg"
echo "  - $NUM_TASKS tasks per trial (first task = healthy/no damage)"
echo "  - Random leg selection (avoiding consecutive same leg)"
echo "  - Damaged leg is LOCKED in bent position (frozen joints)"
echo "  - $PPO_STEPS_PER_TASK steps per task"
echo ""

# ============================================
# 1. PPO (baseline) - GPU 0
# ============================================
(
echo ""
echo "=============================================="
echo "PHASE 1: PPO (baseline) - GPU $GPU_PPO"
echo "=============================================="

for TRIAL in $(seq 1 $NUM_TRIALS); do
    SEED=$((42 + TRIAL - 1))
    
    RUN_NAME="ppo_${ENV}_continual_legdamage_trial${TRIAL}"
    PROJECT_DIR="${BASE_DIR}/ppo_${ENV}_continual_legdamage"
    OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "[GPU $GPU_PPO] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."
    
    LOG_FILE="${OUTPUT_DIR}/train.log"
    python source/mujoco/train_RL_quadruped_continual.py \
        --env $ENV \
        --num_tasks $NUM_TASKS \
        --num_timesteps_per_task $PPO_STEPS_PER_TASK \
        --trial $TRIAL \
        --seed $SEED \
        --cuda_device $GPU_PPO \
        --avoid_consecutive \
        --output_dir $OUTPUT_DIR \
        --experiment_name $RUN_NAME \
        --wandb_project ${WANDB_PROJECT}_ppo \
        > "$LOG_FILE" 2>&1
    
    echo "[GPU $GPU_PPO] Completed: $RUN_NAME"
done

echo ""
echo "PPO experiments complete!"
) &
PPO_PID=$!

# ============================================
# 2. ReDo-PPO - GPU 1
# ============================================
(
echo ""
echo "=============================================="
echo "PHASE 2: ReDo-PPO - GPU $GPU_REDO"
echo "=============================================="

for TRIAL in $(seq 1 $NUM_TRIALS); do
    SEED=$((42 + TRIAL - 1))
    
    RUN_NAME="redo_ppo_${ENV}_continual_legdamage_trial${TRIAL}"
    PROJECT_DIR="${BASE_DIR}/redo_ppo_${ENV}_continual_legdamage"
    OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "[GPU $GPU_REDO] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."
    
    LOG_FILE="${OUTPUT_DIR}/train.log"
    python source/mujoco/train_RL_quadruped_continual.py \
        --env $ENV \
        --num_tasks $NUM_TASKS \
        --num_timesteps_per_task $PPO_STEPS_PER_TASK \
        --trial $TRIAL \
        --seed $SEED \
        --cuda_device $GPU_REDO \
        --avoid_consecutive \
        --output_dir $OUTPUT_DIR \
        --experiment_name $RUN_NAME \
        --use_redo \
        --redo_frequency 1 \
        --wandb_project ${WANDB_PROJECT}_redo \
        > "$LOG_FILE" 2>&1
    
    echo "[GPU $GPU_REDO] Completed: $RUN_NAME"
done

echo ""
echo "ReDo-PPO experiments complete!"
) &
REDO_PID=$!

# ============================================
# 3. TRAC-PPO - GPU 2
# ============================================
(
echo ""
echo "=============================================="
echo "PHASE 3: TRAC-PPO - GPU $GPU_TRAC"
echo "=============================================="

for TRIAL in $(seq 1 $NUM_TRIALS); do
    SEED=$((42 + TRIAL - 1))
    
    RUN_NAME="trac_ppo_${ENV}_continual_legdamage_trial${TRIAL}"
    PROJECT_DIR="${BASE_DIR}/trac_ppo_${ENV}_continual_legdamage"
    OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "[GPU $GPU_TRAC] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."
    
    LOG_FILE="${OUTPUT_DIR}/train.log"
    python source/mujoco/train_RL_quadruped_continual.py \
        --env $ENV \
        --num_tasks $NUM_TASKS \
        --num_timesteps_per_task $PPO_STEPS_PER_TASK \
        --trial $TRIAL \
        --seed $SEED \
        --cuda_device $GPU_TRAC \
        --avoid_consecutive \
        --output_dir $OUTPUT_DIR \
        --experiment_name $RUN_NAME \
        --use_trac \
        --wandb_project ${WANDB_PROJECT}_trac \
        > "$LOG_FILE" 2>&1
    
    echo "[GPU $GPU_TRAC] Completed: $RUN_NAME"
done

echo ""
echo "TRAC-PPO experiments complete!"
) &
TRAC_PID=$!

# ============================================
# WAIT FOR ALL PARALLEL PROCESSES
# ============================================
echo "All three methods started in parallel..."
echo "  PPO (PID: $PPO_PID) on GPU $GPU_PPO"
echo "  ReDo-PPO (PID: $REDO_PID) on GPU $GPU_REDO"
echo "  TRAC-PPO (PID: $TRAC_PID) on GPU $GPU_TRAC"
echo ""
echo "Waiting for all methods to complete..."

wait $PPO_PID $REDO_PID $TRAC_PID

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=============================================="
echo "  PPO: $NUM_TRIALS trials (GPU $GPU_PPO)"
echo "  ReDo-PPO: $NUM_TRIALS trials (GPU $GPU_REDO)"
echo "  TRAC-PPO: $NUM_TRIALS trials (GPU $GPU_TRAC)"
echo "  Total: $((NUM_TRIALS * 3)) runs"
echo "  Output: $BASE_DIR"
echo "=============================================="
