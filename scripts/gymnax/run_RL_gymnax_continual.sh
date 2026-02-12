#!/bin/bash
# Run PPO/TRAC/ReDo on Gymnax environments (Continual)
#
# Usage:
#   ./run_RL_gymnax_continual.sh          # Run all methods on all envs
#   ./run_RL_gymnax_continual.sh ppo      # Run only PPO

set -e

# Configuration
GPU="${GPU:-2}"
NUM_TRIALS="${NUM_TRIALS:-3}"
METHODS="${1:-ppo trac redo}"
ENVS="${ENVS:-CartPole-v1 Acrobot-v1 MountainCar-v0}"

echo "=========================================="
echo "PPO/TRAC/ReDo on Gymnax (Continual)"
echo "=========================================="
echo "GPU: $GPU"
echo "Trials: $NUM_TRIALS"
echo "Methods: $METHODS"
echo "Environments: $ENVS"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TRAIN_SCRIPT="$REPO_ROOT/source/gymnax/train_RL_gymnax_continual.py"

for ENV in $ENVS; do
    for METHOD in $METHODS; do
        echo ""
        echo "=========================================="
        echo "Training $METHOD on $ENV (Continual)"
        echo "=========================================="
        
        for TRIAL in $(seq 1 $NUM_TRIALS); do
            echo ""
            echo "Starting Trial $TRIAL / $NUM_TRIALS ($METHOD on $ENV)"
            echo "----------------------------------------"
            
            CUDA_VISIBLE_DEVICES=$GPU python "$TRAIN_SCRIPT" \
                --env "$ENV" \
                --method "$METHOD" \
                --trial "$TRIAL" \
                --gpus "$GPU" \
                --noise_range 1.0
            
            echo "Trial $TRIAL complete!"
        done
        
        echo ""
        echo "$METHOD on $ENV (Continual): All $NUM_TRIALS trials complete!"
    done
done

echo ""
echo "=========================================="
echo "All continual training complete!"
echo "=========================================="
