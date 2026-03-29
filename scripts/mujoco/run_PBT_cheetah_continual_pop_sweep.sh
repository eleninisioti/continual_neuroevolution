#!/bin/bash
# Run PBT + PPO Continual on MuJoCo CheetahRun for a sweep of population sizes.
# Uses friction continual task (cycles: default 1.0 -> low 0.2 -> high 5.0)
#
# Usage: ./run_PBT_cheetah_continual_pop_sweep.sh [GPU_ID] [NUM_TRIALS]
#
# Environment variables:
#   PBT_MODE: full|hp_only|weights_only (default: weights_only)
#   PBT_INTERVAL: PBT exploit/explore interval in epochs (default: 10)
#   POP_SIZES: space-separated list of population sizes (default: "1 2 4 8 16 32 64 128 256 512")

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============================================
# ACTIVATE VIRTUAL ENVIRONMENT
# ============================================
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
    echo "Activated virtual environment: ${REPO_ROOT}/.venv"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Using active virtual environment: $VIRTUAL_ENV"
else
    echo "WARNING: No virtual environment found. Python commands may fail."
fi

cd "$REPO_ROOT"

# ============================================
# CONFIGURATION
# ============================================
GPU_ID=${1:-0}
NUM_TRIALS=${2:-5}
BASE_SEED=42

WANDB_PROJECT="PBT_popsize_study"
ENV="CheetahRun"

# PBT settings
PBT_MODE="${PBT_MODE:-weights_only}"
PBT_INTERVAL="${PBT_INTERVAL:-10}"

# Continual task settings (friction)
TASK_MOD="friction"
NUM_TASKS=30
TIMESTEPS_PER_TASK=51200000  # 51.2M steps per task
NUM_EVALS_PER_TASK=100

# PPO hyperparameters
NUM_ENVS=256
BATCH_SIZE=256
EPISODE_LENGTH=1000

# Friction multipliers
FRICTION_DEFAULT=1.0
FRICTION_LOW=0.2
FRICTION_HIGH=5.0

# Population sizes to sweep (can be overridden via POP_SIZES env var)
if [ -z "$POP_SIZES" ]; then
    POP_SIZES_ARRAY=(2 4 8 16 32 64 128 256 512)
else
    read -ra POP_SIZES_ARRAY <<< "$POP_SIZES"
fi

# Output directory
BASE_DIR="projects/mujoco"

echo "=========================================="
echo "PBT + PPO Continual - CheetahRun Pop Sweep"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Trials: $NUM_TRIALS"
echo "PBT mode: $PBT_MODE"
echo "PBT interval: $PBT_INTERVAL epochs"
echo "Task mod: $TASK_MOD"
echo "Num tasks: $NUM_TASKS"
echo "Timesteps per task: $TIMESTEPS_PER_TASK"
echo "Pop sizes: ${POP_SIZES_ARRAY[*]}"
echo "=========================================="

mkdir -p "$BASE_DIR"

for pop_size in "${POP_SIZES_ARRAY[@]}"; do
    echo ""
    echo "=========================================="
    echo "PBT Continual | $ENV | pop_size=$pop_size"
    echo "=========================================="

    for trial in $(seq 1 $NUM_TRIALS); do
        SEED=$((BASE_SEED + trial - 1))

        RUN_NAME="pbt_${PBT_MODE}_${ENV}_continual_friction_pop${pop_size}_trial${trial}"
        PROJECT_DIR="${BASE_DIR}/pbt_${PBT_MODE}_${ENV}_continual_friction_pop_sweep/pop_${pop_size}"
        OUTPUT_DIR="${PROJECT_DIR}/trial_${trial}"
        mkdir -p "$OUTPUT_DIR"

        echo ""
        echo ">>> PBT Continual | $ENV | pop=$pop_size | Trial $trial/$NUM_TRIALS | Seed $SEED"

        LOG_FILE="${OUTPUT_DIR}/train.log"
        python source/mujoco/train_PBT_cheetah_continual.py \
            --env $ENV \
            --task_mod $TASK_MOD \
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
            --trial $trial \
            --gpus $GPU_ID \
            --pop_size $pop_size \
            --pbt_mode $PBT_MODE \
            --pbt_interval $PBT_INTERVAL \
            --run_name $RUN_NAME \
            --output_dir $OUTPUT_DIR \
            --wandb_project "$WANDB_PROJECT" \
            2>&1 | tee "$LOG_FILE"

        echo "Trial $trial (pop=$pop_size) complete!"
    done

    echo ""
    echo "Pop size $pop_size: All $NUM_TRIALS trials complete!"
done

echo ""
echo "=========================================="
echo "PBT Continual pop-size sweep complete!"
echo "Results in ${BASE_DIR}/pbt_${PBT_MODE}_${ENV}_continual_friction_pop_sweep/"
echo "=========================================="
