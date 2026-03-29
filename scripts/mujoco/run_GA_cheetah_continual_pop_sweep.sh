#!/bin/bash
# Run continual GA on MuJoCo CheetahRun for a sweep of population sizes.
# Uses friction continual task (cycles: default 1.0 -> low 0.2 -> high 5.0)
#
# Usage: ./run_GA_cheetah_continual_pop_sweep.sh [GPU_ID] [NUM_TRIALS]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$REPO_ROOT"

# ============================================
# CONFIGURATION
# ============================================
GPU_ID=${1:-0}
NUM_TRIALS=${2:-5}
BASE_SEED=42

WANDB_PROJECT="GA_popsize_study"
ENV="CheetahRun"

# Continual task settings (friction)
TASK_MOD="friction"
NUM_TASKS=30
GENS_PER_TASK=500
NUM_EVALS=10  # Evaluations per agent for fitness averaging
FRICTION_DEFAULT=1.0
FRICTION_LOW=0.2
FRICTION_HIGH=5.0

# Population sizes to sweep
POP_SIZES=(2 4 8 16 32 64 128 256 512)

echo "=========================================="
echo "GA Continual - CheetahRun Pop Sweep"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Trials: $NUM_TRIALS"
echo "Task mod: $TASK_MOD"
echo "Num tasks: $NUM_TASKS"
echo "Gens per task: $GENS_PER_TASK"
echo "Pop sizes: ${POP_SIZES[*]}"
echo "=========================================="

for pop_size in "${POP_SIZES[@]}"; do
    for trial in $(seq 1 $NUM_TRIALS); do
        SEED=$((BASE_SEED + trial - 1))
        OUTPUT_DIR="projects/mujoco/ga_${ENV}_continual_pop_sweep/pop_${pop_size}/trial_${trial}"

        echo ""
        echo ">>> GA Continual | $ENV | pop=$pop_size | Trial $trial | Seed $SEED"

        python source/mujoco/train_GA_cheetah_continual.py \
            --env "$ENV" \
            --gpus "$GPU_ID" \
            --pop_size $pop_size \
            --seed $SEED \
            --num_evals $NUM_EVALS \
            --task_mod "$TASK_MOD" \
            --num_tasks $NUM_TASKS \
            --gens_per_task $GENS_PER_TASK \
            --friction_default_mult $FRICTION_DEFAULT \
            --friction_low_mult $FRICTION_LOW \
            --friction_high_mult $FRICTION_HIGH \
            --output_dir "$OUTPUT_DIR" \
            --wandb_project "$WANDB_PROJECT"

        echo "Trial $trial (pop=$pop_size) complete!"
    done
done

echo ""
echo "=========================================="
echo "Continual pop-size sweep complete!"
echo "Results in projects/mujoco/ga_${ENV}_continual_pop_sweep/"
echo "=========================================="
