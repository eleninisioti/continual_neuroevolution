#!/bin/bash
# Run non-continual DNS on Go1 quadruped for all 5 tasks (healthy + 4 leg damages)
# Usage: ./run_DNS_quadruped.sh [GPU_ID] [NUM_TRIALS]

GPU_ID=${1:-4}
NUM_TRIALS=${2:-4}
LEGS="NONE FR FL RR RL"

ENV="Go1JoystickFlatTerrain"
OUTPUT_BASE="projects/mujoco/dns_${ENV}"

echo "=========================================="
echo "DNS Non-Continual - Go1 Leg Damage"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Trials: $NUM_TRIALS"
echo "Tasks: $LEGS (NONE=healthy)"
echo "=========================================="

for leg in $LEGS; do
    echo ""
    echo "=========================================="
    if [ "$leg" = "NONE" ]; then
        echo "Training for HEALTHY robot (no damage)"
    else
        echo "Training for damaged leg=${leg}"
    fi
    echo "=========================================="
    
    for trial in $(seq 1 $NUM_TRIALS); do
        echo ""
        echo "Starting Trial $trial / $NUM_TRIALS (leg=${leg})"
        echo "----------------------------------------"
        
        SEED=$((42 + trial * 1000))
        TRIAL_OUTPUT="${OUTPUT_BASE}_leg${leg}/trial_${trial}"
        
        python source/mujoco/train_DNS_quadruped.py \
            --env "$ENV" \
            --gpus "$GPU_ID" \
            --leg $leg \
            --seed $SEED \
            --trial $trial \
            --output_dir "$TRIAL_OUTPUT" \
            --wandb_project "continual_neuroevolution_dns"
        
        echo "Trial $trial (leg=${leg}) complete!"
    done
done

echo ""
echo "=========================================="
echo "All trials complete!"
echo "=========================================="
