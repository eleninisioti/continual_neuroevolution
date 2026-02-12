#!/bin/bash
# Run all Gymnax experiments: GA, ES, DNS x 3 envs x 3 trials

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$REPO_ROOT"

GPUS="${1:-0}"
WANDB_PROJECT="continual_neuroevolution_gymnax"

ENVS=("CartPole-v1" "Acrobot-v1" "MountainCar-v0")
TRIALS=(1 2 3)
BASE_SEED=42

echo "=========================================="
echo "Running all Gymnax experiments on GPU $GPUS"
echo "Envs: ${ENVS[*]}"
echo "Trials: ${TRIALS[*]}"
echo "=========================================="

# GA experiments
echo ""
echo "========== GA Experiments =========="
for env in "${ENVS[@]}"; do
    for trial in "${TRIALS[@]}"; do
        seed=$((BASE_SEED + trial - 1))
        echo ""
        echo ">>> GA | $env | Trial $trial | Seed $seed"
        python source/gymnax/train_GA_gymnax.py \
            --env "$env" \
            --gpus "$GPUS" \
            --trial "$trial" \
            --seed "$seed" \
            --wandb_project "$WANDB_PROJECT"
    done
done

# ES experiments
echo ""
echo "========== ES Experiments =========="
for env in "${ENVS[@]}"; do
    for trial in "${TRIALS[@]}"; do
        seed=$((BASE_SEED + trial - 1))
        echo ""
        echo ">>> ES | $env | Trial $trial | Seed $seed"
        python source/gymnax/train_ES_gymnax.py \
            --env "$env" \
            --gpus "$GPUS" \
            --trial "$trial" \
            --seed "$seed" \
            --wandb_project "$WANDB_PROJECT"
    done
done

# DNS experiments
echo ""
echo "========== DNS Experiments =========="
for env in "${ENVS[@]}"; do
    for trial in "${TRIALS[@]}"; do
        seed=$((BASE_SEED + trial - 1))
        echo ""
        echo ">>> DNS | $env | Trial $trial | Seed $seed"
        python source/gymnax/train_DNS_gymnax.py \
            --env "$env" \
            --gpus "$GPUS" \
            --trial "$trial" \
            --seed "$seed" \
            --wandb_project "$WANDB_PROJECT"
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
