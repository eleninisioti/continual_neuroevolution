#!/bin/bash
# Run TRAC-PPO and ReDo-PPO on Go1 quadruped for all 5 tasks (healthy + 4 leg damages)
# Usage: ./run_RL_quadruped_variants.sh [GPU_ID] [NUM_TRIALS]

GPU_ID=${1:-3}
NUM_TRIALS=${2:-3}
LEGS="NONE FR FL RR RL"

ENV="Go1JoystickFlatTerrain"

echo "=========================================="
echo "PPO Variants (TRAC/ReDo) - Go1 Leg Damage"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Trials: $NUM_TRIALS"
echo "Tasks: $LEGS (NONE=healthy)"
echo "=========================================="

# Run TRAC-PPO for all legs
echo ""
echo "=========================================="
echo "TRAC-PPO Training"
echo "=========================================="

for leg in $LEGS; do
    echo ""
    echo "=========================================="
    if [ "$leg" = "NONE" ]; then
        echo "TRAC-PPO: HEALTHY robot (no damage)"
    else
        echo "TRAC-PPO: damaged leg=${leg}"
    fi
    echo "=========================================="
    
    for trial in $(seq 1 $NUM_TRIALS); do
        echo ""
        echo "Starting Trial $trial / $NUM_TRIALS (leg=${leg})"
        echo "----------------------------------------"
        
        SEED=$((42 + trial * 1000))
        OUTPUT_DIR="projects/mujoco/trac_ppo_${ENV}_leg${leg}/trial_${trial}"
        
        python source/mujoco/train_RL_quadruped.py \
            --env "$ENV" \
            --gpus "$GPU_ID" \
            --leg $leg \
            --seed $SEED \
            --trial $trial \
            --use_trac \
            --output_dir "$OUTPUT_DIR" \
            --wandb_project "continual_neuroevolution_ppo"
        
        echo "TRAC-PPO Trial $trial (leg=${leg}) complete!"
    done
done

# Run ReDo-PPO for all legs
echo ""
echo "=========================================="
echo "ReDo-PPO Training"
echo "=========================================="

for leg in $LEGS; do
    echo ""
    echo "=========================================="
    if [ "$leg" = "NONE" ]; then
        echo "ReDo-PPO: HEALTHY robot (no damage)"
    else
        echo "ReDo-PPO: damaged leg=${leg}"
    fi
    echo "=========================================="
    
    for trial in $(seq 1 $NUM_TRIALS); do
        echo ""
        echo "Starting Trial $trial / $NUM_TRIALS (leg=${leg})"
        echo "----------------------------------------"
        
        SEED=$((42 + trial * 1000))
        OUTPUT_DIR="projects/mujoco/redo_ppo_${ENV}_leg${leg}/trial_${trial}"
        
        python source/mujoco/train_RL_quadruped.py \
            --env "$ENV" \
            --gpus "$GPU_ID" \
            --leg $leg \
            --seed $SEED \
            --trial $trial \
            --use_redo \
            --output_dir "$OUTPUT_DIR" \
            --wandb_project "continual_neuroevolution_ppo"
        
        echo "ReDo-PPO Trial $trial (leg=${leg}) complete!"
    done
done

echo ""
echo "=========================================="
echo "All TRAC-PPO and ReDo-PPO trials complete!"
echo "=========================================="
