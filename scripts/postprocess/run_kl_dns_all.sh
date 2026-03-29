#!/bin/bash
# Compute KL divergence for all DNS continual gymnax experiments.
# Runs compute_kl_gymnax.py for each env × pop_size × trial.
#
# Usage:
#   bash scripts/postprocess/run_kl_dns_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

source .venv/bin/activate

ENVS=("CartPole-v1" "MountainCar-v0" "Acrobot-v1")

for env in "${ENVS[@]}"; do
    env_dir="projects/gymnax/dns_continual_pop_sweep/$env"
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

            if [ -f "$trial_dir/kl_divergence.json" ]; then
                echo "Skipping DNS $env $pop $trial (already computed)"
                continue
            fi

            echo ""
            echo ">>> DNS | $env | $pop | $trial"
            python scripts/postprocess/compute_kl_gymnax.py \
                --project_dir "$trial_dir" \
                --method dns \
                --env "$env"
        done
    done
done

echo ""
echo "DNS KL computation done!"
