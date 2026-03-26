#!/bin/bash

# Script to run checkpoint evaluation for all environments under ga (continual and vanilla)
# Runs for all trials (default: 0-4) with 50 evaluation episodes per trial
# Usage: ./run_eval_all_envs.sh [--min_gen MIN] [--max_gen MAX] [--num_episodes NUM] [--trial_start START] [--trial_end END] [--gpu GPU_ID]

# Default values
MIN_GEN=100
MAX_GEN=2000
NUM_EPISODES=50
TRIAL_START=0
TRIAL_END=4  # Run for trials 0-4 (5 trials total)
GPU_ID=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --min_gen)
            MIN_GEN="$2"
            shift 2
            ;;
        --max_gen)
            MAX_GEN="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --trial_start)
            TRIAL_START="$2"
            shift 2
            ;;
        --trial_end)
            TRIAL_END="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--min_gen MIN] [--max_gen MAX] [--num_episodes NUM] [--trial_start START] [--trial_end END] [--gpu GPU_ID]"
            exit 1
            ;;
    esac
done

# Base directory
BASE_DIR="projects/reports/iclr_rebuttal/kl_divergence/ga"

# Variants to evaluate
VARIANTS=("continual" "vanilla")

# Environments to evaluate
ENVS=("acrobot" "cartpole" "mountaincar")

# Activate virtual environment
source .venv/bin/activate

# Run evaluation for each variant, environment, and trial
for variant in "${VARIANTS[@]}"; do
    for env in "${ENVS[@]}"; do
        for trial in $(seq $TRIAL_START $TRIAL_END); do
            PROJECT_DIR="${BASE_DIR}/${variant}/${env}"
            
            echo "=========================================="
            echo "Running evaluation for: $variant/$env (trial $trial)"
            echo "Project directory: $PROJECT_DIR"
            echo "Trial: $trial"
            echo "Generation range: $MIN_GEN to $MAX_GEN"
            echo "Number of episodes: $NUM_EPISODES"
            echo "=========================================="
            
            # Check if project directory exists
            if [ ! -d "$PROJECT_DIR" ]; then
                echo "Warning: Project directory not found: $PROJECT_DIR"
                echo "Skipping $variant/$env (trial $trial)..."
                echo ""
                continue
            fi
            
            # Check if trial directory exists
            TRIAL_DIR="${PROJECT_DIR}/trial_${trial}"
            if [ ! -d "$TRIAL_DIR" ]; then
                echo "Warning: Trial directory not found: $TRIAL_DIR"
                echo "Skipping $variant/$env (trial $trial)..."
                echo ""
                continue
            fi
            
            # Run the evaluation script on specified GPU
            echo "Using GPU: $GPU_ID"
            CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/postprocess/eval_checkpoints.py \
                --project_dir "$PROJECT_DIR" \
                --trial $trial \
                --min_gen $MIN_GEN \
                --max_gen $MAX_GEN \
                --num_episodes $NUM_EPISODES \
                --force_rerun
            
            # Check exit status
            if [ $? -eq 0 ]; then
                echo "✓ Successfully completed evaluation for $variant/$env (trial $trial)"
            else
                echo "✗ Error occurred while evaluating $variant/$env (trial $trial)"
            fi
            
            echo ""
            echo "----------------------------------------"
            echo ""
        done
    done
done

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="

