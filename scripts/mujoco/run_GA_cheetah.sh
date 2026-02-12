#!/bin/bash
# Run non-continual GA on CheetahRun for multiple friction values
# Usage: ./run_GA_cheetah.sh [GPU_ID] [NUM_TRIALS]

GPU_ID=${1:-7}
NUM_TRIALS=${2:-5}
FRICTIONS="5.0"

ENV="CheetahRun"
OUTPUT_BASE="projects/mujoco/ga_${ENV}"

echo "=========================================="
echo "GA Non-Continual - CheetahRun Friction"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Trials: $NUM_TRIALS"
echo "Friction values: $FRICTIONS"
echo "=========================================="

for friction in $FRICTIONS; do
    echo ""
    echo "=========================================="
    echo "Training for friction=${friction}"
    echo "=========================================="
    
    # Convert friction to label (e.g., 0.5 -> friction0p5)
    friction_label=$(echo $friction | sed 's/\./_/')
    
    for trial in $(seq 1 $NUM_TRIALS); do
        echo ""
        echo "Starting Trial $trial / $NUM_TRIALS (friction=${friction})"
        echo "----------------------------------------"
        
        SEED=$((42 + trial * 1000))
        TRIAL_OUTPUT="${OUTPUT_BASE}_friction${friction_label}/trial_${trial}"
        
        python source/mujoco/train_GA_cheetah.py \
            --env "$ENV" \
            --gpus "$GPU_ID" \
            --friction $friction \
            --seed $SEED \
            --trial $trial \
            --output_dir "$TRIAL_OUTPUT" \
            --wandb_project "continual_neuroevolution_ga"
        
        echo "Trial $trial (friction=${friction}) complete!"
    done
done

echo ""
echo "=========================================="
echo "All trials complete!"
echo "=========================================="
