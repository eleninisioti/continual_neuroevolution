#!/bin/bash
# Run PBT CheetahRun continual friction training - 10 trials
# Runs trials sequentially on a single GPU
#
# Usage: ./run_PBT_cheetah_continual.sh [GPU_ID]
#   GPU_ID: GPU to use (default: 0)
#
# Environment variables:
#   PBT_MODE: full|hp_only|weights_only (default: full)
#   PBT_INTERVAL: PBT exploit/explore interval in generations (default: 10)
#   NUM_TRIALS: number of trials to run (default: 10)

set -e

GPU_ID=${1:-0}
PBT_MODE=${PBT_MODE:-full}
PBT_INTERVAL=${PBT_INTERVAL:-10}
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

# Environment
ENV="CheetahRun"

# Task settings
NUM_TASKS=30
TIMESTEPS_PER_TASK=51200000  # 51.2M steps per task (matches RL script)
NUM_EVALS_PER_TASK=100       # 100 eval points per task

# PPO hyperparameters (defaults matching RL script for CheetahRun)
NUM_ENVS=256
BATCH_SIZE=256
EPISODE_LENGTH=1000

# Friction multipliers
FRICTION_DEFAULT=1.0
FRICTION_LOW=0.2
FRICTION_HIGH=5.0

# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================
mkdir -p "$BASE_DIR"

echo "=============================================="
echo "PBT (${PBT_MODE}) CheetahRun Continual Friction"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Trials: $NUM_TRIALS"
echo "PBT mode: $PBT_MODE"
echo "PBT interval: $PBT_INTERVAL generations"
echo ""
echo "Experiment: Friction changes between tasks"
echo "  - $NUM_TASKS tasks per trial"
echo "  - Friction cycles: default ($FRICTION_DEFAULT) -> low ($FRICTION_LOW) -> high ($FRICTION_HIGH)"
echo "  - $TIMESTEPS_PER_TASK timesteps per task, $NUM_EVALS_PER_TASK evals per task"
echo ""

# Run trials sequentially
for TRIAL in $(seq 1 $NUM_TRIALS); do
    SEED=$((42 + TRIAL - 1))

    RUN_NAME="pbt_${PBT_MODE}_${ENV}_continual_friction_trial${TRIAL}"
    PROJECT_DIR="${BASE_DIR}/pbt_${PBT_MODE}_${ENV}_continual_friction"
    OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
    mkdir -p "$OUTPUT_DIR"

    echo "[GPU $GPU_ID] Starting: $RUN_NAME (trial $TRIAL/$NUM_TRIALS, seed=$SEED)..."

    LOG_FILE="${OUTPUT_DIR}/train.log"
    python source/mujoco/train_PBT_cheetah_continual.py \
        --env $ENV \
        --task_mod friction \
        --friction_default_mult $FRICTION_DEFAULT \
        --friction_low_mult $FRICTION_LOW \
        --friction_high_mult $FRICTION_HIGH \
        --num_tasks $NUM_TASKS \
        --timesteps_per_task $TIMESTEPS_PER_TASK \
        --num_evals_per_task $NUM_EVALS_PER_TASK \
        --num_envs $NUM_ENVS \
        --batch_size $BATCH_SIZE \
        --episode_length $EPISODE_LENGTH \
        --seed $SEED \
        --trial $TRIAL \
        --gpus $GPU_ID \
        --pop_size 1 \
        --pbt_mode $PBT_MODE \
        --pbt_interval $PBT_INTERVAL \
        --run_name $RUN_NAME \
        --output_dir $OUTPUT_DIR \
        --wandb_project ${WANDB_PROJECT}_pbt \
        2>&1 | tee "$LOG_FILE"

    echo "[GPU $GPU_ID] Completed: $RUN_NAME"
done

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=============================================="
echo "  PBT (${PBT_MODE}): $NUM_TRIALS trials"
echo "  Output: $BASE_DIR"
echo "=============================================="
