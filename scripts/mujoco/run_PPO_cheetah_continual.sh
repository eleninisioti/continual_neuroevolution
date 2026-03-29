#!/bin/bash
# Run PPO CheetahRun continual friction training - 10 trials
# Usage: ./run_PPO_cheetah_continual.sh [GPU_ID]

set -e

GPU_ID=${1:-0}
NUM_TRIALS=${NUM_TRIALS:-10}

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

ENV="CheetahRun"
PPO_STEPS_PER_TASK=51200000
NUM_TASKS=30
NUM_ENVS=256
NUM_EVALS_PER_TASK=100

FRICTION_DEFAULT=1.0
FRICTION_LOW=0.2
FRICTION_HIGH=5.0

# ============================================
mkdir -p "$BASE_DIR"

echo "=============================================="
echo "PPO CheetahRun Continual Friction"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Trials: $NUM_TRIALS"
echo "Tasks: $NUM_TASKS"
echo "Steps per task: $PPO_STEPS_PER_TASK"
echo "Num envs: $NUM_ENVS"
echo "Evals per task: $NUM_EVALS_PER_TASK"
echo ""

for TRIAL in $(seq 1 $NUM_TRIALS); do
    SEED=$((42 + TRIAL))

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
        --num_envs $NUM_ENVS \
        --num_evals_per_task $NUM_EVALS_PER_TASK \
        --seed $SEED \
        --gpus $GPU_ID \
        --run_name $RUN_NAME \
        --output_dir $OUTPUT_DIR \
        --wandb_project ${WANDB_PROJECT}_ppo \
        2>&1 | tee "$LOG_FILE"

    echo "[GPU $GPU_ID] Completed: $RUN_NAME"
done

echo ""
echo "=============================================="
echo "ALL PPO EXPERIMENTS COMPLETE!"
echo "=============================================="
echo "  Trials: $NUM_TRIALS"
echo "  Output: $BASE_DIR/ppo_${ENV}_continual_friction/"
echo "=============================================="
