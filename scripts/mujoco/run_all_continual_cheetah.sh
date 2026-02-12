#!/bin/bash
# Run GA, ES, and DNS continual experiments for CheetahRun with num_evals=3
# All algorithms run sequentially on the same GPU
#
# Usage: ./run_all_continual_cheetah.sh [GPU_ID] [NUM_TRIALS]

set -e

GPU_ID=${1:-0}
NUM_TRIALS=${2:-10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [ -f "${REPO_DIR}/.venv/bin/activate" ]; then
    source "${REPO_DIR}/.venv/bin/activate"
    echo "Activated virtual environment: ${REPO_DIR}/.venv"
fi

cd "$REPO_DIR"

# Configuration
WANDB_PROJECT="continual_neuroevolution_ICML_2026"
BASE_DIR="projects/mujoco"
ENV="CheetahRun"
NUM_TASKS=30
NUM_EVALS=3

# Friction multipliers
FRICTION_DEFAULT=1.0
FRICTION_LOW=0.2
FRICTION_HIGH=5.0

echo "=============================================="
echo "Running GA, ES, DNS Continual - CheetahRun"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Trials: $NUM_TRIALS"
echo "num_evals: $NUM_EVALS"
echo "=============================================="

mkdir -p "$BASE_DIR"

# ============================================
# GA Continual
# ============================================
echo ""
echo "=============================================="
echo "Starting GA Continual ($NUM_TRIALS trials)"
echo "=============================================="

GENS_PER_TASK=500
for TRIAL in $(seq 1 $NUM_TRIALS); do
    SEED=$((42 + TRIAL - 1))
    RUN_NAME="ga_${ENV}_continual_friction_trial${TRIAL}"
    PROJECT_DIR="${BASE_DIR}/ga_${ENV}_continual_friction"
    OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "[GA] Starting trial $TRIAL/$NUM_TRIALS (seed=$SEED)..."
    
    python source/mujoco/train_GA_cheetah_continual.py \
        --env $ENV \
        --task_mod friction \
        --friction_default_mult $FRICTION_DEFAULT \
        --friction_low_mult $FRICTION_LOW \
        --friction_high_mult $FRICTION_HIGH \
        --num_tasks $NUM_TASKS \
        --gens_per_task $GENS_PER_TASK \
        --num_evals $NUM_EVALS \
        --seed $SEED \
        --gpus $GPU_ID \
        --pop_size 512 \
        --run_name $RUN_NAME \
        --output_dir $OUTPUT_DIR \
        --wandb_project ${WANDB_PROJECT}_ga \
        > "${OUTPUT_DIR}/train.log" 2>&1
    
    echo "[GA] Completed trial $TRIAL"
done

# ============================================
# ES Continual
# ============================================
echo ""
echo "=============================================="
echo "Starting ES Continual ($NUM_TRIALS trials)"
echo "=============================================="

GENS_PER_TASK=500
for TRIAL in $(seq 1 $NUM_TRIALS); do
    SEED=$((42 + TRIAL - 1))
    RUN_NAME="es_${ENV}_continual_friction_trial${TRIAL}"
    PROJECT_DIR="${BASE_DIR}/es_${ENV}_continual_friction"
    OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "[ES] Starting trial $TRIAL/$NUM_TRIALS (seed=$SEED)..."
    
    python source/mujoco/train_ES_cheetah_continual.py \
        --env $ENV \
        --task_mod friction \
        --friction_default_mult $FRICTION_DEFAULT \
        --friction_low_mult $FRICTION_LOW \
        --friction_high_mult $FRICTION_HIGH \
        --num_tasks $NUM_TASKS \
        --gens_per_task $GENS_PER_TASK \
        --num_evals $NUM_EVALS \
        --seed $SEED \
        --gpus $GPU_ID \
        --pop_size 512 \
        --run_name $RUN_NAME \
        --output_dir $OUTPUT_DIR \
        --wandb_project ${WANDB_PROJECT}_es \
        > "${OUTPUT_DIR}/train.log" 2>&1
    
    echo "[ES] Completed trial $TRIAL"
done

# ============================================
# DNS Continual
# ============================================
echo ""
echo "=============================================="
echo "Starting DNS Continual ($NUM_TRIALS trials)"
echo "=============================================="

EPISODES_PER_TASK=200
for TRIAL in $(seq 1 $NUM_TRIALS); do
    SEED=$((42 + TRIAL - 1))
    RUN_NAME="dns_${ENV}_continual_friction_trial${TRIAL}"
    PROJECT_DIR="${BASE_DIR}/dns_${ENV}_continual_friction"
    OUTPUT_DIR="${PROJECT_DIR}/trial_${TRIAL}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "[DNS] Starting trial $TRIAL/$NUM_TRIALS (seed=$SEED)..."
    
    python source/mujoco/train_DNS_cheetah_continual.py \
        --env $ENV \
        --task_mod friction \
        --friction_default_mult $FRICTION_DEFAULT \
        --friction_low_mult $FRICTION_LOW \
        --friction_high_mult $FRICTION_HIGH \
        --num_tasks $NUM_TASKS \
        --episodes_per_task $EPISODES_PER_TASK \
        --num_evals $NUM_EVALS \
        --seed $SEED \
        --gpus $GPU_ID \
        --pop_size 512 \
        --run_name $RUN_NAME \
        --output_dir $OUTPUT_DIR \
        --wandb_project ${WANDB_PROJECT}_dns \
        > "${OUTPUT_DIR}/train.log" 2>&1
    
    echo "[DNS] Completed trial $TRIAL"
done

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=============================================="
echo "  GA: $NUM_TRIALS trials"
echo "  ES: $NUM_TRIALS trials"
echo "  DNS: $NUM_TRIALS trials"
echo "=============================================="
