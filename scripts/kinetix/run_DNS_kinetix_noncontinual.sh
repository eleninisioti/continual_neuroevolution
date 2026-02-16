#!/bin/bash

# Non-continual DNS (Dominated Novelty Search) training on 20 medium Kinetix tasks (h0-h19)
# Custom DNS implementation: isoline variation + novelty-based selection.
#
# Usage:
#   ./run_DNS_kinetix_noncontinual.sh                  # Run all envs on GPU 0
#   ./run_DNS_kinetix_noncontinual.sh --cuda 3         # Run on GPU 3
#   ./run_DNS_kinetix_noncontinual.sh --env h0_unicycle  # Run single env
#   ./run_DNS_kinetix_noncontinual.sh --num_trials 5   # Run 5 trials
#   ./run_DNS_kinetix_noncontinual.sh --popsize 512    # Custom population size

set -e

# Default settings
GPU=4
NUM_TRIALS=1
POPSIZE=256
BATCH_SIZE=128
GENERATIONS=200
ISO_SIGMA=0.05
LINE_SIGMA=0.5
K=3
NUM_EVALS=3
SEED=0
ENV=""
WANDB_PROJECT="Kinetix-noncontinual-dns"
NO_WANDB=""
EVAL_BATCH_SIZE=128
EPISODE_LENGTH=1000
LOG_INTERVAL=10

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
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --generations)
            GENERATIONS="$2"
            shift 2
            ;;
        --iso_sigma)
            ISO_SIGMA="$2"
            shift 2
            ;;
        --line_sigma)
            LINE_SIGMA="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --num_evals)
            NUM_EVALS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --eval_batch_size)
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --episode_length)
            EPISODE_LENGTH="$2"
            shift 2
            ;;
        --log_interval)
            LOG_INTERVAL="$2"
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
            echo "Usage: $0 [--cuda GPU] [--num_trials N] [--env ENV] [--popsize N] [--batch_size N] [--generations N] [--iso_sigma F] [--line_sigma F] [--k K] [--num_evals N] [--seed S] [--no_wandb]"
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
echo "Kinetix DNS Non-Continual Training"
echo "========================================"
echo "GPU: $GPU"
echo "Num trials: $NUM_TRIALS"
echo "Population size: $POPSIZE"
echo "Batch size (offspring): $BATCH_SIZE"
echo "Generations: $GENERATIONS"
echo "iso_sigma: $ISO_SIGMA"
echo "line_sigma: $LINE_SIGMA"
echo "k (novelty neighbors): $K"
echo "Num evals: $NUM_EVALS"
echo "Episode length: $EPISODE_LENGTH"
echo "Eval batch size: $EVAL_BATCH_SIZE"
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
CMD="python experiments/dns.py"
CMD="$CMD --gpu $GPU"
CMD="$CMD --pop_size $POPSIZE"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --num_generations $GENERATIONS"
CMD="$CMD --iso_sigma $ISO_SIGMA"
CMD="$CMD --line_sigma $LINE_SIGMA"
CMD="$CMD --k $K"
CMD="$CMD --num_evals $NUM_EVALS"
CMD="$CMD --episode_length $EPISODE_LENGTH"
CMD="$CMD --eval_batch_size $EVAL_BATCH_SIZE"
CMD="$CMD --log_interval $LOG_INTERVAL"
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
echo "Starting DNS training..."
echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "========================================"
echo "All DNS training completed!"
echo "========================================"
