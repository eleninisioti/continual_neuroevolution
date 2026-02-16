#!/bin/bash

# Non-continual OpenES training on 20 medium Kinetix tasks (h0-h19)
# Uses actor-only network with evosax Open_ES ask/tell pattern.
#
# Usage:
#   ./run_ES_kinetix_noncontinual.sh                  # Run all envs on GPU 0
#   ./run_ES_kinetix_noncontinual.sh --cuda 3         # Run on GPU 3
#   ./run_ES_kinetix_noncontinual.sh --env h0_unicycle  # Run single env
#   ./run_ES_kinetix_noncontinual.sh --num_trials 5   # Run 5 trials
#   ./run_ES_kinetix_noncontinual.sh --popsize 512    # Custom population size

set -e

# Default settings
GPU=3
NUM_TRIALS=1
POPSIZE=1024
GENERATIONS=200
SIGMA=0.5
LEARNING_RATE=0.01
SEED=0
ENV=""
WANDB_PROJECT="Kinetix-noncontinual-es"
NO_WANDB=""

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
        --sigma)
            SIGMA="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
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
        --no_wandb)
            NO_WANDB="--no_wandb"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cuda GPU] [--num_trials N] [--env ENV] [--popsize N] [--generations N] [--sigma F] [--learning_rate F] [--seed S] [--no_wandb]"
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
echo "Kinetix OpenES Non-Continual Training"
echo "========================================"
echo "GPU: $GPU"
echo "Num trials: $NUM_TRIALS"
echo "Population size: $POPSIZE"
echo "Generations: $GENERATIONS"
echo "Sigma: $SIGMA"
echo "Learning rate: $LEARNING_RATE"
echo "Seed: $SEED"
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

# Activate venv if it exists
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
    echo "Using venv: $REPO_ROOT/.venv"
fi

# Build command
CMD="python experiments/es.py"
CMD="$CMD --gpu $GPU"
CMD="$CMD --popsize $POPSIZE"
CMD="$CMD --generations $GENERATIONS"
CMD="$CMD --sigma $SIGMA"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --seed $SEED"
CMD="$CMD --num_trials $NUM_TRIALS"
CMD="$CMD --wandb_project $WANDB_PROJECT"
CMD="$CMD --project_dir $PROJECT_DIR"
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
echo "All OpenES training completed!"
echo "========================================"
