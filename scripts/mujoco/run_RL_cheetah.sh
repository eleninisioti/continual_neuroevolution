#!/bin/bash
# Run non-continual PPO on CheetahRun for all 3 friction values
# Usage: ./run_RL_cheetah.sh [GPU_ID] [NUM_TRIALS]

GPU_ID=${1:-3}
NUM_TRIALS=${2:-3}
FRICTIONS="0.2 1.0 5.0"

ENV="CheetahRun"
OUTPUT_BASE="projects/mujoco/ppo_${ENV}"

echo "=========================================="
echo "PPO Non-Continual - CheetahRun Friction"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Trials: $NUM_TRIALS"
echo "Friction values: $FRICTIONS"
echo "=========================================="

for friction in $FRICTIONS; do
    friction_label=$(echo "friction${friction}" | sed 's/\./_/g')
    
    echo ""
    echo "=========================================="
    echo "Training for friction=${friction}"
    echo "=========================================="
    
    for trial in $(seq 1 $NUM_TRIALS); do
        echo ""
        echo "Starting Trial $trial / $NUM_TRIALS (friction=${friction})"
        echo "----------------------------------------"
        
        SEED=$((42 + trial * 1000))
        TRIAL_OUTPUT="${OUTPUT_BASE}_friction${friction}/trial_${trial}"
        
        python source/mujoco/train_RL_cheetah.py \
            --env "$ENV" \
            --gpus "$GPU_ID" \
            --friction $friction \
            --seed $SEED \
            --trial $trial \
            --output_dir "$TRIAL_OUTPUT" \
            --wandb_project "continual_neuroevolution_ppo"
        
        echo "Trial $trial (friction=${friction}) complete!"
    done
done

echo ""
echo "=========================================="
echo "All trials complete!"
echo "=========================================="
