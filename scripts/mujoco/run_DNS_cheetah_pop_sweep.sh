#!/bin/bash
# Run non-continual DNS on CheetahRun with different population sizes
# Usage: ./run_DNS_cheetah_pop_sweep.sh [GPU_ID] [NUM_TRIALS]

GPU_ID=${1:-0}
NUM_TRIALS=${2:-5}
FRICTION="1.0"
#POP_SIZES=(1 2 4 8 16 32 64 128 256 512)
POP_SIZES=(512 256 128 64 32 16 8 4 2 1)
ENV="CheetahRun"
WANDB_PROJECT="DNS_popsize_study"
BASE_SEED=42

echo "=========================================="
echo "DNS Non-Continual - CheetahRun Pop Sweep"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Trials: $NUM_TRIALS"
echo "Friction: $FRICTION"
echo "Pop sizes: ${POP_SIZES[*]}"
echo "=========================================="

for pop_size in "${POP_SIZES[@]}"; do
    # Batch size should be at most pop_size/2
    if [ $pop_size -lt 4 ]; then
        batch_size=$pop_size
    else
        batch_size=$((pop_size / 2))
    fi

    for trial in $(seq 1 $NUM_TRIALS); do
        SEED=$((BASE_SEED + trial - 1))
        OUTPUT_DIR="projects/mujoco/dns_${ENV}_pop_sweep/pop_${pop_size}/trial_${trial}"

        echo ""
        echo ">>> pop_size=$pop_size (batch=$batch_size) | Trial $trial / $NUM_TRIALS"

        python source/mujoco/train_DNS_cheetah.py \
            --env "$ENV" \
            --gpus "$GPU_ID" \
            --friction $FRICTION \
            --pop_size $pop_size \
            --batch_size $batch_size \
            --seed $SEED \
            --trial $trial \
            --output_dir "$OUTPUT_DIR" \
            --wandb_project "$WANDB_PROJECT"

        echo "Trial $trial (pop=$pop_size) complete!"
    done
done

echo ""
echo "=========================================="
echo "Pop sweep complete!"
echo "Results in projects/mujoco/dns_${ENV}_pop_sweep/"
echo "=========================================="
