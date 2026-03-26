#!/bin/bash

# Non-continual GA (SimpleGA) training on 20 medium Kinetix tasks (h0-h19)
# Uses actor-only network with evosax SimpleGA ask/tell pattern.
#
# Usage:
#   ./run_GA_kinetix_noncontinual.sh                  # Run all envs on GPU 0
#   ./run_GA_kinetix_noncontinual.sh --cuda 3         # Run on GPU 3
#   ./run_GA_kinetix_noncontinual.sh --env h0_unicycle  # Run single env
#   ./run_GA_kinetix_noncontinual.sh --num_trials 5   # Run 5 trials
#   ./run_GA_kinetix_noncontinual.sh --popsize 512    # Custom population size

set -e

# Default settings
GPU=0
NUM_TRIALS=10
POPSIZE=1024
GENERATIONS=200
SIGMA_INIT=0.001
SEED=0
ENV=""
WANDB_PROJECT="Kinetix-noncontinual-ga"
NO_WANDB=""
EVAL_REPS=10
EVOLVE_REPS=10

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
        --env)
            ENV="$2"
            shift 2
            ;;
        --popsize)
            POPSIZE="$2"
            shift 2
            ;;
        --generations)
            GENERATIONS="$2"
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
            echo "Usage: $0 [--cuda GPU] [--num_trials N] [--env ENV] [--popsize N] [--generations N] [--sigma_init F] [--seed S] [--eval_reps N] [--evolve_reps N] [--no_wandb]"
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
echo "Kinetix SimpleGA Non-Continual Training"
echo "========================================"
echo "GPU: $GPU"
echo "Num trials: $NUM_TRIALS"
echo "Population size: $POPSIZE"
echo "Generations: $GENERATIONS"
echo "Sigma init: $SIGMA_INIT"
echo "Seed: $SEED"
echo "Eval reps: $EVAL_REPS"
echo "Evolve reps: $EVOLVE_REPS"
if [ -n "$ENV" ]; then
    echo "Environment: $ENV"
else
    echo "Environment: ALL (20 medium tasks)"
fi
echo "Working directory: $KINETIX_DIR"
echo "Project directory: $PROJECT_DIR"
echo "========================================"

# Create projects directory
mkdir -p "$PROJECT_DIR"

cd "$KINETIX_DIR"

# Activate virtualenv: prefer `uv` if present, else fall back to `.venv`
if [ -f "$REPO_ROOT/uv/bin/activate" ]; then
    source "$REPO_ROOT/uv/bin/activate"
    echo "Using virtualenv: $REPO_ROOT/uv"
elif [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
    echo "Using venv: $REPO_ROOT/.venv"
fi

# Build command
CMD="python experiments/ga.py"
CMD="$CMD --gpu $GPU"
CMD="$CMD --popsize $POPSIZE"
CMD="$CMD --generations $GENERATIONS"
CMD="$CMD --sigma_init $SIGMA_INIT"
CMD="$CMD --seed $SEED"
CMD="$CMD --num_trials $NUM_TRIALS"
CMD="$CMD --wandb_project $WANDB_PROJECT"
CMD="$CMD --project_dir $PROJECT_DIR"
CMD="$CMD --eval_reps $EVAL_REPS"
CMD="$CMD --evolve_reps $EVOLVE_REPS"
if [ -n "$ENV" ]; then
    CMD="$CMD --env $ENV"
fi
if [ -n "$NO_WANDB" ]; then
    CMD="$CMD --no_wandb"
fi

echo ""
echo "Starting training..."
echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "========================================"
echo "All SimpleGA training completed!"
echo "========================================"
