#!/bin/bash
# Run OpenES non-continual CheetahRun friction experiments
# 3 conditions (0.2x, 1.0x, 5.0x) x 10 trials = 30 runs
#
# Usage:
#   ./run_openes_cheetah_noncontinual.sh [GPU_ID]
#   Example: ./run_openes_cheetah_noncontinual.sh 5

set -e

GPU=${1:-5}

echo "=============================================="
echo "Running OpenES Non-Continual CheetahRun on GPU $GPU"
echo "=============================================="

# ============================================
# NON-CONTINUAL CHEETAH FRICTION (3 conditions x 10 trials = 30 runs)
# ============================================
echo ""
echo "=== NON-CONTINUAL CheetahRun Friction (3 conditions x 10 trials) ==="

for friction_mult in 0.2 1.0 5.0; do
    echo ""
    echo "--- Friction multiplier: ${friction_mult}x ---"
    for trial in {1..10}; do
        seed=$((42 + trial - 1))
        echo "Starting CheetahRun friction=${friction_mult}x trial $trial (seed=$seed)..."
        
        # For non-continual, train on single friction for full duration
        python train_openes_continual.py \
            --env CheetahRun \
            --task_mod friction \
            --num_tasks 1 \
            --gens_per_task 15000 \
            --friction_default_mult $friction_mult \
            --friction_low_mult $friction_mult \
            --friction_high_mult $friction_mult \
            --pop_size 512 \
            --sigma 0.04 \
            --learning_rate 0.01 \
            --seed $seed \
            --gpus $GPU \
            --output_dir "projects/openes_CheetahRun_noncont_friction_mult${friction_mult}_trial${trial}" \
            --run_name "openes_CheetahRun_noncont_friction${friction_mult}_seed${seed}" \
            --wandb_project mujoco_evosax
        echo "Completed CheetahRun friction=${friction_mult}x trial $trial"
    done
done

echo ""
echo "=============================================="
echo "ALL OpenES Non-Continual CheetahRun EXPERIMENTS COMPLETED!"
echo "=============================================="
echo ""
echo "Summary:"
echo "  - Non-continual CheetahRun Friction: 30 trials (3 conditions x 10)"
