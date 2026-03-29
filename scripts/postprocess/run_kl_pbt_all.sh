#!/bin/bash
# Compute KL divergence for all PBT continual gymnax experiments.
# Runs compute_kl_gymnax.py for each env × pop_size × trial.
#
# Usage:
#   bash scripts/postprocess/run_kl_pbt_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

source .venv/bin/activate

declare -A ENV_DIRS
ENV_DIRS["CartPole-v1"]="pbt_weights_only_CartPole_v1_continual"
ENV_DIRS["MountainCar-v0"]="pbt_weights_only_MountainCar_v0_continual"
ENV_DIRS["Acrobot-v1"]="pbt_weights_only_Acrobot_v1_continual"

for env in "CartPole-v1" "MountainCar-v0" "Acrobot-v1"; do
    env_dir="projects/gymnax/${ENV_DIRS[$env]}"
    if [ ! -d "$env_dir" ]; then
        echo "Skipping $env: $env_dir not found"
        continue
    fi

    for pop_dir in "$env_dir"/pop_*/; do
        [ -d "$pop_dir" ] || continue
        pop=$(basename "$pop_dir")

        for trial_dir in "$pop_dir"/trial_*/; do
            [ -d "$trial_dir" ] || continue
            trial=$(basename "$trial_dir")

            # Skip if already computed
            if [ -f "$trial_dir/kl_divergence.json" ]; then
                echo "Skipping $env $pop $trial (already computed)"
                continue
            fi

            echo ""
            echo ">>> $env | $pop | $trial"
            python scripts/postprocess/compute_kl_gymnax.py \
                --project_dir "$trial_dir" \
                --method pbt \
                --env "$env"
        done
    done
done

echo ""
echo "All done!"
