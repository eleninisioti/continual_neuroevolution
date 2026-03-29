#!/bin/bash

# Compute KL divergence between consecutive tasks for gymnax continual runs.
# Runs for all methods, environments, and trials.
#
# Usage:
#   ./scripts/postprocess/run_kl_gymnax.sh [--methods METHOD1,METHOD2] [--envs ENV1,ENV2] \
#       [--trial_start START] [--trial_end END] [--num_episodes N] [--gpu GPU_ID]

# Default values
METHODS="pbt"
ENVS="CartPole-v1,Acrobot-v1,MountainCar-v0"
TRIAL_START=1
TRIAL_END=1
NUM_EPISODES=10
GPU_ID=0

# PBT modes to check (only relevant for method=pbt)
PBT_MODES="full"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --envs)
            ENVS="$2"
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
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --pbt_modes)
            PBT_MODES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--methods pbt,ga] [--envs CartPole-v1,...] [--trial_start N] [--trial_end N] [--num_episodes N] [--gpu GPU_ID] [--pbt_modes full,weights_only]"
            exit 1
            ;;
    esac
done

IFS=',' read -ra METHOD_LIST <<< "$METHODS"
IFS=',' read -ra ENV_LIST <<< "$ENVS"
IFS=',' read -ra PBT_MODE_LIST <<< "$PBT_MODES"

echo "=========================================="
echo "KL Divergence Analysis for Gymnax Continual"
echo "=========================================="
echo "Methods: ${METHOD_LIST[*]}"
echo "Environments: ${ENV_LIST[*]}"
echo "Trials: $TRIAL_START to $TRIAL_END"
echo "Num episodes: $NUM_EPISODES"
echo "GPU: $GPU_ID"
echo "=========================================="

for method in "${METHOD_LIST[@]}"; do
    for env in "${ENV_LIST[@]}"; do
        # Build list of project dirs to process for this method+env
        if [ "$method" = "pbt" ]; then
            # PBT uses underscores in env name: CartPole_v1
            env_underscore="${env//-/_}"
            DIRS=()
            for mode in "${PBT_MODE_LIST[@]}"; do
                DIRS+=("pbt_${mode}_${env_underscore}_continual")
            done
        elif [ "$method" = "ga" ]; then
            # GA uses original env name: CartPole-v1
            DIRS=("ga_${env}_continual")
        else
            echo "Unknown method: $method, skipping"
            continue
        fi

        for dir_name in "${DIRS[@]}"; do
            for trial in $(seq $TRIAL_START $TRIAL_END); do
                PROJECT_DIR="projects/gymnax/${dir_name}/trial_${trial}"

                echo ""
                echo "=========================================="
                echo "Method: $method | Env: $env | Dir: $dir_name | Trial: $trial"
                echo "Project: $PROJECT_DIR"
                echo "=========================================="

                # Check if project directory exists
                if [ ! -d "$PROJECT_DIR" ]; then
                    echo "Warning: Directory not found: $PROJECT_DIR — skipping"
                    continue
                fi

                # Check if checkpoints exist
                if [ ! -d "${PROJECT_DIR}/checkpoints" ]; then
                    echo "Warning: No checkpoints/ dir in $PROJECT_DIR — skipping"
                    echo "  (Re-run training with updated script to save per-task checkpoints)"
                    continue
                fi

                CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/postprocess/compute_kl_gymnax.py \
                    --project_dir "$PROJECT_DIR" \
                    --method "$method" \
                    --env "$env" \
                    --num_episodes "$NUM_EPISODES"

                if [ $? -eq 0 ]; then
                    echo "OK: ${dir_name}/trial_${trial}"
                else
                    echo "FAILED: ${dir_name}/trial_${trial}"
                fi
            done
        done
    done
done

echo ""
echo "=========================================="
echo "All KL divergence analyses completed!"
echo "=========================================="
