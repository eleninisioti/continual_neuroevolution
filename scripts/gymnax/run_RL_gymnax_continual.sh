#!/bin/bash
# Run PPO/TRAC/ReDo on Gymnax environments (Continual)
#
# Usage:
#   ./run_RL_gymnax_continual.sh          # Run all methods on all envs (parallel on GPUs 0,1,2)
#   ./run_RL_gymnax_continual.sh ppo      # Run only PPO on GPU 0

set -e

# Configuration
NUM_TRIALS="${NUM_TRIALS:-1}"
METHODS="${1:-ppo trac redo}"
ENVS="${ENVS:- MountainCar-v0}"

# Map methods to GPUs
declare -A METHOD_GPU
METHOD_GPU[ppo]=7
METHOD_GPU[trac]=7
METHOD_GPU[redo]=7

echo "=========================================="
echo "PPO/TRAC/ReDo on Gymnax (Continual)"
echo "=========================================="
echo "Trials: $NUM_TRIALS"
echo "Methods: $METHODS"
echo "Environments: $ENVS"
echo "GPU mapping: ppo->0, trac->1, redo->2"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TRAIN_SCRIPT="$REPO_ROOT/source/gymnax/train_RL_gymnax_continual.py"

for ENV in $ENVS; do
    echo ""
    echo "=========================================="
    echo "Training on $ENV (Continual) - All methods in parallel"
    echo "=========================================="
    
    PIDS=()
    
    for METHOD in $METHODS; do
        GPU=${METHOD_GPU[$METHOD]}
        
        for TRIAL in $(seq 1 $NUM_TRIALS); do
            echo "Starting $METHOD on GPU $GPU (Trial $TRIAL)"
            
            CUDA_VISIBLE_DEVICES=$GPU python "$TRAIN_SCRIPT" \
                --env "$ENV" \
                --method "$METHOD" \
                --trial "$TRIAL" \
                --gpus "$GPU" \
                --noise_range 1.0 &
            
            PIDS+=($!)
        done
    done
    
    # Wait for all methods to complete for this environment
    echo "Waiting for all methods on $ENV to complete..."
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    
    echo "$ENV: All methods complete!"
done

echo ""
echo "=========================================="
echo "All continual training complete!"
echo "=========================================="
