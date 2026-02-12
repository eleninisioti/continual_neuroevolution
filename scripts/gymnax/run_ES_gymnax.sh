#!/bin/bash
# Run ES on Gymnax environments
# Usage: ./run_ES_gymnax.sh [GPU]
#   Runs 3 trials on CartPole, Acrobot, MountainCar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$REPO_ROOT"

GPUS="${1:-0}"
WANDB_PROJECT="continual_neuroevolution_gymnax"
BASE_SEED=42

ENVS=("CartPole-v1" "Acrobot-v1" "MountainCar-v0")
TRIALS=(1 2 3)

echo "Running ES on all Gymnax envs (3 trials each) on GPU $GPUS"

for env in "${ENVS[@]}"; do
    for trial in "${TRIALS[@]}"; do
        seed=$((BASE_SEED + trial - 1))
        echo ""
        echo ">>> ES | $env | Trial $trial | Seed $seed"
        python source/gymnax/train_ES_gymnax.py --env "$env" --gpus "$GPUS" --trial "$trial" --seed "$seed" --wandb_project "$WANDB_PROJECT"
    done
done

echo "ES experiments completed!"
