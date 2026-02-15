#!/bin/bash

# Continual GA (SimpleGA) training on 20 medium Kinetix tasks (h0-h19)
# Population carries over between tasks – no reset.
# Each task gets --generations_per_task generations (default 200).
#
# Usage:
#   ./run_GA_kinetix_continual.sh                          # Run on GPU 0
#   ./run_GA_kinetix_continual.sh --cuda 3                 # Run on GPU 3
#   ./run_GA_kinetix_continual.sh --generations_per_task 100
#   ./run_GA_kinetix_continual.sh --num_trials 5
#   ./run_GA_kinetix_continual.sh --no_wandb

set -e

# Default settings
GPU=2
NUM_TRIALS=1
POPSIZE=1024
GENERATIONS_PER_TASK=200
SIGMA_INIT=0.001
SEED=0
WANDB_PROJECT="Kinetix-continual-ga"
NO_WANDB=""
EVAL_REPS=10
EVOLVE_REPS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            GPU="$2"
            shift 2
            ;;
        --num_trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --popsize)
            POPSIZE="$2"
            shift 2
            ;;
        --generations_per_task)
            GENERATIONS_PER_TASK="$2"
            shift 2
            ;;
        --sigma_init)
            SIGMA_INIT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --eval_reps)
            EVAL_REPS="$2"
            shift 2
            ;;
        --evolve_reps)
            EVOLVE_REPS="$2"
            shift 2
            ;;
        --no_wandb)
            NO_WANDB="--no_wandb"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cuda GPU] [--num_trials N] [--popsize N] [--generations_per_task N] [--sigma_init F] [--seed S] [--eval_reps N] [--evolve_reps N] [--no_wandb]"
            exit 1
            ;;
    esac
done

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KINETIX_DIR="$REPO_ROOT/source/kinetix"
PROJECT_DIR="$REPO_ROOT/projects/kinetix"

echo "========================================"
echo "Kinetix SimpleGA Continual Training"
echo "========================================"
echo "GPU: $GPU"
echo "Num trials: $NUM_TRIALS"
echo "Population size: $POPSIZE"
echo "Generations per task: $GENERATIONS_PER_TASK"
echo "Sigma init: $SIGMA_INIT"
echo "Seed: $SEED"
echo "Eval reps: $EVAL_REPS"
echo "Evolve reps: $EVOLVE_REPS"
echo "Working directory: $KINETIX_DIR"
echo "Project directory: $PROJECT_DIR"
echo "========================================"

# Create projects directory
mkdir -p "$PROJECT_DIR"

cd "$KINETIX_DIR"

# Activate venv if it exists
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
    echo "Using venv: $REPO_ROOT/.venv"
fi

# Build command
CMD="python experiments/ga_continual.py"
CMD="$CMD --gpu $GPU"
CMD="$CMD --popsize $POPSIZE"
CMD="$CMD --generations_per_task $GENERATIONS_PER_TASK"
CMD="$CMD --sigma_init $SIGMA_INIT"
CMD="$CMD --seed $SEED"
CMD="$CMD --num_trials $NUM_TRIALS"
CMD="$CMD --wandb_project $WANDB_PROJECT"
CMD="$CMD --project_dir $PROJECT_DIR"
CMD="$CMD --eval_reps $EVAL_REPS"
CMD="$CMD --evolve_reps $EVOLVE_REPS"
if [ -n "$NO_WANDB" ]; then
    CMD="$CMD --no_wandb"
fi

echo ""
echo "Starting continual training..."
echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "========================================"
echo "All SimpleGA continual training completed!"
echo "========================================"
