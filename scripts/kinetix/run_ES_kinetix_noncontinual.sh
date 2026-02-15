#!/bin/bash

# Non-continual OpenES training on 20 medium Kinetix tasks (h0-h19)
# Uses pixel-based observations by default
#
# Usage:
#   ./run_ES_kinetix_noncontinual.sh                  # Run on GPU 0
#   ./run_ES_kinetix_noncontinual.sh --cuda 3         # Run on GPU 3
#   ./run_ES_kinetix_noncontinual.sh --symbolic       # Run with symbolic observations
#   ./run_ES_kinetix_noncontinual.sh --seed 42        # Run with specific seed

set -e

# Default settings
GPU=0
ENV_CONFIG="pixels"  # pixels or symbolic
SEEDS="0"
WANDB_PROJECT="Kinetix-ES-noncontinual"

# 20 medium h-tasks
ENVIRONMENTS=(
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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            GPU="$2"
            shift 2
            ;;
        --symbolic)
            ENV_CONFIG="symbolic"
            shift
            ;;
        --seed)
            SEEDS="$2"
            shift 2
            ;;
        --project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cuda GPU] [--symbolic] [--seed SEED] [--project PROJECT]"
            exit 1
            ;;
    esac
done

# Get the directory of this script and navigate to source/kinetix
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KINETIX_DIR="$(cd "$SCRIPT_DIR/../../source/kinetix" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_DIR="$REPO_ROOT/projects/kinetix"

echo "========================================"
echo "Kinetix OpenES Non-Continual Training"
echo "========================================"
echo "GPU: $GPU"
echo "Observation type: $ENV_CONFIG"
echo "Seeds: $SEEDS"
echo "Wandb project: $WANDB_PROJECT"
echo "Kinetix directory: $KINETIX_DIR"
echo "Project directory: $PROJECT_DIR"
echo "========================================"

# Create projects directory
mkdir -p "$PROJECT_DIR"
mkdir -p "$KINETIX_DIR/data/outputs_es_noncontinual"

cd "$KINETIX_DIR"

# Activate kinetix-specific venv if it exists
if [ -f "$REPO_ROOT/.venv-kinetix/bin/activate" ]; then
    source "$REPO_ROOT/.venv-kinetix/bin/activate"
    echo "Using kinetix venv: $REPO_ROOT/.venv-kinetix"
fi

# Build symbolic flag
SYMBOLIC_FLAG=""
if [[ "$ENV_CONFIG" == "symbolic" ]]; then
    SYMBOLIC_FLAG="--symbolic"
fi

for env_name in "${ENVIRONMENTS[@]}"; do
    echo ""
    echo "========================================" 
    echo "Training ES on: $env_name"
    echo "========================================"
    
    # Convert seed index to trial index (1-based)
    trial_idx=1
    for seed in $SEEDS; do
        echo "  Seed: $seed (trial $trial_idx)"
        
        CUDA_VISIBLE_DEVICES=$GPU python3 train_ES_kinetix.py \
            --env "$env_name" \
            $SYMBOLIC_FLAG \
            --seed $seed \
            --trial $trial_idx \
            --gpus "$GPU" \
            --wandb_project "$WANDB_PROJECT" \
            --project_dir "$PROJECT_DIR" \
            >> "data/outputs_es_noncontinual/${env_name}_seed${seed}.txt" 2>&1
        
        echo "    Done (see data/outputs_es_noncontinual/${env_name}_seed${seed}.txt)"
        ((trial_idx++))
    done
done

echo ""
echo "========================================"
echo "All ES training completed!"
echo "========================================"
