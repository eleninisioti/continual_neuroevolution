#!/bin/bash
# Run GA on Gymnax environments for a sweep of population sizes.
# Usage: ./run_GA_gymnax_pop_sweep.sh [GPU]
#   Runs 5 trials × 9 pop sizes × 3 envs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$REPO_ROOT"

GPUS="${1:-0}"
WANDB_PROJECT="GA_popsize_study"
BASE_SEED=42

ENVS=("CartPole-v1" "Acrobot-v1" "MountainCar-v0")
#POP_SIZES=(1 2 4 8 16 32 128 256 512)
POP_SIZES=(64)
TRIALS=(1 2 3 4 5)

echo "Running GA pop-size sweep on all Gymnax envs (5 trials each) on GPU $GPUS"
echo "Pop sizes: ${POP_SIZES[*]}"

for env in "${ENVS[@]}"; do
    for pop_size in "${POP_SIZES[@]}"; do
        for trial in "${TRIALS[@]}"; do
            seed=$((BASE_SEED + trial - 1))
            # Use a separate output dir per pop_size so results don't overwrite
            OUTPUT_DIR="projects/gymnax/ga_pop_sweep/${env}/pop_${pop_size}/trial_${trial}"
            echo ""
            echo ">>> GA | $env | pop=$pop_size | Trial $trial | Seed $seed"
            python source/gymnax/train_GA_gymnax.py \
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
echo "Pop-size sweep completed!"
echo "Results in projects/gymnax/ga_pop_sweep/"
