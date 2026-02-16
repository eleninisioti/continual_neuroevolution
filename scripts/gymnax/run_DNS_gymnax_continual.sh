#!/bin/bash
# Run DNS gymnax continual experiments across 3 envs and 3 trials

ENVS=("CartPole-v1" "Acrobot-v1" "MountainCar-v0")
TRIALS=(1)
GPU="7"

for env in "${ENVS[@]}"; do
    for trial in "${TRIALS[@]}"; do
        echo "========================================"
        echo "DNS Continual: $env, Trial $trial"
        echo "========================================"
        python source/gymnax/train_DNS_gymnax_continual.py --env "$env" --trial "$trial" --gpus "$GPU" --noise_range 1.0
    done
done

echo "All DNS continual experiments complete!"
