#!/bin/bash
# Run DNS continual learning on Go1 quadruped with leg damage
# Usage: ./run_DNS_quadruped_continual.sh [GPU_ID] [NUM_TRIALS]

# Default values
GPU_ID=${1:-7}
NUM_TRIALS=${2:-10}
NUM_TASKS=20
EPISODES_PER_TASK=50

ENV="Go1JoystickFlatTerrain"
OUTPUT_BASE="projects/mujoco/dns_${ENV}_continual_legdamage"

echo "=========================================="
echo "DNS Continual Learning - Go1 Leg Damage"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Trials: $NUM_TRIALS"
echo "Tasks per trial: $NUM_TASKS"
echo "Episodes per task: $EPISODES_PER_TASK"
echo "Output: $OUTPUT_BASE"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Run trials
for trial in $(seq 1 $NUM_TRIALS); do
    echo ""
    echo "Starting Trial $trial / $NUM_TRIALS"
    echo "----------------------------------------"
    
    SEED=$((42 + trial * 1000))
    TRIAL_OUTPUT="${OUTPUT_BASE}/trial_${trial}"
    
    python source/mujoco/train_DNS_quadruped_continual.py \
        --env "$ENV" \
        --gpus "$GPU_ID" \
        --num_tasks $NUM_TASKS \
        --episodes_per_task $EPISODES_PER_TASK \
        --pop_size 512 \
        --batch_size 256 \
        --k 3 \
        --seed $SEED \
        --trial $trial \
        --output_dir "$TRIAL_OUTPUT" \
        --experiment_name "dns_go1_legdamage_trial${trial}" \
        --wandb_project "continual_neuroevolution_dns" \
        --avoid_consecutive
    
    echo "Trial $trial complete!"
done

echo ""
echo "=========================================="
echo "All $NUM_TRIALS trials complete!"
echo "Results saved to: $OUTPUT_BASE"
echo "=========================================="
