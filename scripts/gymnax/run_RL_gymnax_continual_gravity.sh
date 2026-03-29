#!/bin/bash
# Run PPO/TRAC/ReDo on Gymnax envs with parameter-based continual tasks
#   CartPole: vary gravity [0.98, 98.0] (factor of 10, matching TRAC paper)
#   MountainCar: vary gravity [0.00025, 0.025] (factor of 10)
#   Acrobot: vary link_length_1 [0.5, 2.0]
#
# Usage:
#   ./run_RL_gymnax_continual_gravity.sh              # all methods, all envs
#   ./run_RL_gymnax_continual_gravity.sh ppo           # PPO only
#   ENVS="CartPole-v1" ./run_RL_gymnax_continual_gravity.sh  # one env

set -e

GPU="${GPU:-3}"
NUM_TRIALS="${NUM_TRIALS:-5}"
ENVS="${ENVS:- CartPole-v1 }"
METHODS="${1:-trac redo}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "PPO/TRAC/ReDo on Gymnax (Continual - Param variation)"
echo "=========================================="
echo "Trials: $NUM_TRIALS"
echo "Methods: $METHODS"
echo "Envs: $ENVS"
echo "GPU: $GPU"
echo "=========================================="

for env in $ENVS; do
    for method in $METHODS; do
        for trial in $(seq 1 $NUM_TRIALS); do
            OUTPUT_DIR="projects/gymnax/rl_continual_gravity_sweep/${env}/${method}/trial_${trial}"
            echo ""
            echo ">>> $env | $method | Trial $trial / $NUM_TRIALS -> $OUTPUT_DIR"
            CUDA_VISIBLE_DEVICES=$GPU python "$REPO_ROOT/source/gymnax/train_RL_gymnax_continual.py" \
                --env "$env" \
                --method "$method" \
                --task_type param \
                --trial "$trial" \
                --gpus "$GPU" \
                --output_dir "$OUTPUT_DIR"
        done
    done
done

echo ""
echo "All runs complete!"
