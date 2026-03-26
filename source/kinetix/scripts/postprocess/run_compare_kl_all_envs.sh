#!/bin/bash

# Script to run KL-divergence comparison for all environments under kl_divergence
# Runs for all trials (default: 0-4) 
# Usage: ./run_compare_kl_all_envs.sh [--min_gen MIN] [--max_gen MAX] [--trial_start START] [--trial_end END] [--gpu GPU_ID] [--skip_state_collection]

# Default values
MIN_GEN=100
MAX_GEN=2000
TRIAL_START=0
TRIAL_END=4  # Run for trials 0-4 (5 trials total)
GPU_ID=7
SKIP_STATE_COLLECTION=false

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
        --skip_state_collection)
            SKIP_STATE_COLLECTION=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--min_gen MIN] [--max_gen MAX] [--trial_start START] [--trial_end END] [--gpu GPU_ID] [--skip_state_collection]"
            exit 1
            ;;
    esac
done

# Base directories to process
BASE_DIRS=("ga" "openes")

# Variants to compare
VARIANTS=("continual" "vanilla")

# Environments to compare
ENVS=("acrobot" "cartpole" "mountaincar")

# Activate virtual environment
source .venv/bin/activate

# Run KL-divergence comparison for each base directory, variant, environment, and trial
for base_dir in "${BASE_DIRS[@]}"; do
    BASE_DIR="projects/reports/iclr_rebuttal/kl_divergence/${base_dir}"
    
    for variant in "${VARIANTS[@]}"; do
        for env in "${ENVS[@]}"; do
            for trial in $(seq $TRIAL_START $TRIAL_END); do
                PROJECT_DIR="${BASE_DIR}/${variant}/${env}"
                
                echo "=========================================="
                echo "Running KL-divergence comparison for: ${base_dir}/${variant}/${env} (trial $trial)"
                echo "Project directory: $PROJECT_DIR"
                echo "Trial: $trial"
                echo "Generation range: $MIN_GEN to $MAX_GEN"
                echo "=========================================="
                
                # Check if project directory exists
                if [ ! -d "$PROJECT_DIR" ]; then
                    echo "Warning: Project directory not found: $PROJECT_DIR"
                    echo "Skipping ${base_dir}/${variant}/${env} (trial $trial)..."
                    echo ""
                    continue
                fi
                
                # Check if config.yaml exists
                if [ ! -f "${PROJECT_DIR}/config.yaml" ]; then
                    echo "Warning: config.yaml not found in $PROJECT_DIR"
                    echo "Skipping ${base_dir}/${variant}/${env} (trial $trial)..."
                    echo ""
                    continue
                fi
                
                # Check if trial directory exists
                TRIAL_DIR="${PROJECT_DIR}/trial_${trial}"
                if [ ! -d "$TRIAL_DIR" ]; then
                    echo "Warning: Trial directory not found: $TRIAL_DIR"
                    echo "Skipping ${base_dir}/${variant}/${env} (trial $trial)..."
                    echo ""
                    continue
                fi
                
                # First, run eval_checkpoints.py to collect states and compute action probabilities (if not skipped)
                if [ "$SKIP_STATE_COLLECTION" = false ]; then
                    echo "Step 1: Collecting states and computing action probabilities..."
                    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/postprocess/eval_checkpoints.py \
                        --project_dir "$PROJECT_DIR" \
                        --trial $trial \
                        --min_gen $MIN_GEN \
                        --max_gen $MAX_GEN \
                        --num_episodes 1
                    
                    if [ $? -ne 0 ]; then
                        echo "✗ Error occurred while collecting states for ${base_dir}/${variant}/${env} (trial $trial)"
                        echo "Skipping KL-divergence comparison..."
                        echo ""
                        continue
                    fi
                    echo "✓ State collection completed for ${base_dir}/${variant}/${env} (trial $trial)"
                    echo ""
                else
                    echo "Skipping state collection (using existing data)..."
                fi
                
                # Run the KL-divergence comparison script on specified GPU
                echo "Step 2: Computing KL-divergence between consecutive checkpoints..."
                echo "Using GPU: $GPU_ID"
                CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/postprocess/compare_checkpoints_kl.py \
                    --project_dir "$PROJECT_DIR" \
                    --trial $trial \
                    --min_gen $MIN_GEN \
                    --max_gen $MAX_GEN
                
                # Check exit status
                if [ $? -eq 0 ]; then
                    echo "✓ Successfully completed KL-divergence comparison for ${base_dir}/${variant}/${env} (trial $trial)"
                else
                    echo "✗ Error occurred while comparing ${base_dir}/${variant}/${env} (trial $trial)"
                fi
                
                echo ""
                echo "----------------------------------------"
                echo ""
            done
        done
    done
done

echo "=========================================="
echo "All KL-divergence comparisons completed!"
echo "=========================================="

