#!/bin/bash
#
# Run SimpleGA Non-Continual on Kinetix 20 Medium H-Tasks
#
# Usage:
#   ./run_GA_kinetix_noncontinual.sh [gpu_id] [trial]
#
# Example:
#   ./run_GA_kinetix_noncontinual.sh 0 1
#

set -e

# Defaults
GPU=${1:-5}
TRIAL=${2:-1}

# Path setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_DIR="$REPO_ROOT/source/kinetix"
PROJECT_DIR="$REPO_ROOT/projects/kinetix"

# 20 medium h-tasks
ENVS=(
    "h0_unicycle"
    "h1_car_left"
    "h2_car_right"
    "h3_car_thrust"
    "h4_thrust_the_needle"
    "h5_angry_birds"
    "h6_thrust_over"
    "h7_car_flip"
    "h8_weird_vehicle"
    "h9_spin_the_right_way"
    "h10_thrust_right_easy"
    "h11_thrust_left_easy"
    "h12_thrustfall_left"
    "h13_thrustfall_right"
    "h14_thrustblock"
    "h15_thrustshoot"
    "h16_thrustcontrol_right"
    "h17_thrustcontrol_left"
    "h18_thrust_right_very_easy"
    "h19_thrust_left_very_easy"
)

echo "=============================================="
echo "SimpleGA Non-Continual on Kinetix (20 H-Tasks)"
echo "=============================================="
echo "GPU: $GPU"
echo "Trial: $TRIAL"
echo "=============================================="

for ENV in "${ENVS[@]}"; do
    echo ""
    echo ">>> Training GA on: $ENV"
    echo "-------------------------------------------"
    
    OUTPUT_DIR="$PROJECT_DIR/vanilla/ga/${ENV}/trial_${TRIAL}"
    
    python "$SOURCE_DIR/train_GA_kinetix.py" \
        --env "$ENV" \
        --gpus "$GPU" \
        --trial "$TRIAL" \
        --output_dir "$OUTPUT_DIR" \
        --wandb_project "Kinetix-GA-noncontinual"
    
    echo ">>> Completed: $ENV"
done

echo ""
echo "=============================================="
echo "All tasks completed!"
echo "=============================================="
