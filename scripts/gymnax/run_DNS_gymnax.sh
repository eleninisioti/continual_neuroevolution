#!/bin/bash
# Run DNS on Gymnax environments
# Usage: ./run_DNS_gymnax.sh [GPU]
#   Runs 3 trials on CartPole, Acrobot, MountainCar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$REPO_ROOT"

GPUS="${1:-2}"
WANDB_PROJECT="continual_neuroevolution_gymnax"
BASE_SEED=42

ENVS=("CartPole-v1" "Acrobot-v1" "MountainCar-v0")
TRIALS=(1 2 3)

echo "Running DNS on all Gymnax envs (3 trials each) on GPU $GPUS"

for env in "${ENVS[@]}"; do
    for trial in "${TRIALS[@]}"; do
        seed=$((BASE_SEED + trial - 1))
        echo ""
        echo ">>> DNS | $env | Trial $trial | Seed $seed"
        python source/gymnax/train_DNS_gymnax.py --env "$env" --gpus "$GPUS" --trial "$trial" --seed "$seed" --wandb_project "$WANDB_PROJECT"
    done
done

echo "DNS experiments completed!"
