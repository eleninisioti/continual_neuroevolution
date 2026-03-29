#!/bin/bash
# Run non-continual GA on CheetahRun with different population sizes
# Usage: ./run_GA_cheetah_pop_sweep.sh [GPU_ID] [NUM_TRIALS]

GPU_ID=${1:-7}
NUM_TRIALS=${2:-5}
FRICTION="1.0"
#POP_SIZES=(1 2 4 8 16 32 128 256 512)
POP_SIZES=(64)

ENV="CheetahRun"

echo "=========================================="
echo "GA Non-Continual - CheetahRun Pop Sweep"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Trials: $NUM_TRIALS"
echo "Friction: $FRICTION"
echo "Pop sizes: ${POP_SIZES[*]}"
echo "=========================================="

friction_label=$(echo $FRICTION | sed 's/\./_/')

for pop_size in "${POP_SIZES[@]}"; do
    for trial in $(seq 1 $NUM_TRIALS); do
        echo ""
        echo ">>> pop_size=$pop_size | Trial $trial / $NUM_TRIALS"

        SEED=$((42 + trial * 1000))
        TRIAL_OUTPUT="projects/mujoco/ga_${ENV}_pop_sweep/pop_${pop_size}/trial_${trial}"

        python source/mujoco/train_GA_cheetah.py \
            --env "$ENV" \
            --gpus "$GPU_ID" \
            --friction $FRICTION \
            --pop_size $pop_size \
            --seed $SEED \
            --trial $trial \
            --output_dir "$TRIAL_OUTPUT" \
            --wandb_project "continual_neuroevolution_ga"

        echo "Trial $trial (pop=$pop_size) complete!"
    done
done

echo ""
echo "=========================================="
echo "Pop sweep complete!"
echo "Results in projects/mujoco/ga_${ENV}_pop_sweep/"
echo "=========================================="
