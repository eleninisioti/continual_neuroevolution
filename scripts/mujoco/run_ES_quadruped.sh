#!/bin/bash
# Run OpenES Go1 leg damage (non-continual) - 20 trials across GPUs 0-1
# 5 trials per leg (FR, FL, RR, RL)
# Each trial trains from scratch on a single damaged leg

# Array of available GPUs
GPUS=(4)
NUM_GPUS=${#GPUS[@]}

# Legs: 0=FR, 1=FL, 2=RR, 3=RL
LEGS=(0 1 2 3)
LEG_NAMES=("FR" "FL" "RR" "RL")
TRIALS_PER_LEG=5

echo "=============================================="
echo "OpenES Go1 Leg Damage (Non-Continual) Training"
echo "Running 20 trials across ${NUM_GPUS} GPUs"
echo "=============================================="
echo ""
echo "Experiment: Train from scratch with one damaged leg"
echo "  - 4 legs × 5 trials = 20 total trials"
echo "  - Damaged leg is LOCKED in bent position (frozen joints)"
echo "  - 500 generations per trial"
echo ""

source .venv/bin/activate

# Create output directory
mkdir -p projects

# Track PIDs for each GPU
declare -A GPU_PIDS

# Function to wait for a specific GPU to be free
wait_for_gpu() {
    local gpu=$1
    if [[ -n "${GPU_PIDS[$gpu]}" ]]; then
        echo "Waiting for GPU $gpu (PID ${GPU_PIDS[$gpu]}) to finish..."
        wait ${GPU_PIDS[$gpu]}
        echo "GPU $gpu is now free."
    fi
}

# Run trials: cycle through legs and trials
trial_num=0
for leg in "${LEGS[@]}"; do
    leg_name="${LEG_NAMES[$leg]}"
    
    for t in $(seq 1 $TRIALS_PER_LEG); do
        trial_num=$((trial_num + 1))
        gpu_idx=$(((trial_num - 1) % NUM_GPUS))
        gpu=${GPUS[$gpu_idx]}
        
        # Wait for this GPU to be free before starting
        wait_for_gpu $gpu
        
        echo "Starting trial $trial_num (Leg: $leg_name, Trial $t/$TRIALS_PER_LEG) on GPU $gpu..."
        
        python train_openes_go1_legdamage.py \
            --env Go1JoystickFlatTerrain \
            --damaged_leg $leg \
            --num_generations 500 \
            --trial $t \
            --seed $((42 + trial_num)) \
            --gpus $gpu \
            --output_dir projects \
            --experiment_name "openes_go1_legdamage_${leg_name}_trial${t}" \
            > "projects/openes_go1_legdamage_${leg_name}_trial${t}.log" 2>&1 &
        
        # Store PID for this GPU
        GPU_PIDS[$gpu]=$!
        echo "  Started with PID ${GPU_PIDS[$gpu]}"
        
        # Small delay before starting next
        sleep 2
    done
done

echo ""
echo "All trials launched! Waiting for remaining trials to complete..."
echo "Monitor with: tail -f projects/openes_go1_legdamage_*.log"
echo ""

# Wait for all remaining background jobs
wait
echo "All 20 OpenES trials complete!"
