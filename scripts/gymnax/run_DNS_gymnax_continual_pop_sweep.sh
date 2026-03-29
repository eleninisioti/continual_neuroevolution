#!/bin/bash
# Run continual DNS on Gymnax environments for a sweep of population sizes.
# Usage: ./run_DNS_gymnax_continual_pop_sweep.sh [GPU]
#   Runs 5 trials × 9 pop sizes × 3 envs (continual: 10 tasks each)
#   Envs run in order: Acrobot first, then CartPole, then MountainCar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$REPO_ROOT"

GPUS="${1:-2}"
WANDB_PROJECT="GA_popsize_study"
BASE_SEED=42

ENVS=("MountainCar-v0")
POP_SIZES=(64 128 256 512)
#POP_SIZES=(1)

TRIALS=(1 2 3 4 5)

echo "Running continual DNS pop-size sweep on all Gymnax envs (5 trials each) on GPU $GPUS"
echo "Pop sizes: ${POP_SIZES[*]}"
echo "Env order: ${ENVS[*]}"

for env in "${ENVS[@]}"; do
    for pop_size in "${POP_SIZES[@]}"; do
        for trial in "${TRIALS[@]}"; do
            seed=$((BASE_SEED + trial - 1))
            OUTPUT_DIR="projects/gymnax/dns_continual_pop_sweep/${env}/pop_${pop_size}/trial_${trial}"
            echo ""
            echo ">>> DNS Continual | $env | pop=$pop_size | Trial $trial | Seed $seed"
            python source/gymnax/train_DNS_gymnax_continual.py \
                --env "$env" \
                --gpus "$GPUS" \
                --trial "$trial" \
                --seed "$seed" \
                --pop_size "$pop_size" \
                --output_dir "$OUTPUT_DIR" \
                --wandb_project "$WANDB_PROJECT"
        done
    done
done

echo ""
echo "Continual DNS pop-size sweep completed!"
echo "Results in projects/gymnax/dns_continual_pop_sweep/"
