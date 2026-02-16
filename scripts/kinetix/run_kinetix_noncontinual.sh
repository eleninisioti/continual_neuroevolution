#!/bin/bash

# Non-continual PPO training on 20 medium Kinetix tasks (h0-h19)
# Runs 3 variants: Vanilla PPO, Redo PPO, TRAC PPO
# Uses pixel-based observations by default
#
# Usage:
#   ./run_kinetix_noncontinual.sh                  # Run all 3 variants with pixels on GPU 0
#   ./run_kinetix_noncontinual.sh --cuda 3         # Run on GPU 3
#   ./run_kinetix_noncontinual.sh --symbolic       # Run with symbolic observations
#   ./run_kinetix_noncontinual.sh --seed 42        # Single seed instead of 0-9
#   ./run_kinetix_noncontinual.sh --variant vanilla  # Run only vanilla PPO

# NOTE: We do NOT use 'set -e' here so that a failure on one
# environment does not kill the entire script (and skip remaining envs/variants).

# Default settings
GPU=4
CONFIG_NAME="ppo"
ENV_CONFIG="pixels"  # Default to pixels
SEEDS="0"
WANDB_PROJECT_BASE="Kinetix-noncontinual"
VARIANT="all"  # all, vanilla, redo, or trac

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
            WANDB_PROJECT_BASE="$2"
            shift 2
            ;;
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cuda GPU] [--symbolic] [--seed SEED] [--project PROJECT] [--variant vanilla|redo|trac|all]"
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
echo "Kinetix Non-Continual Training"
echo "========================================"
echo "GPU: $GPU"
echo "Observation type: $ENV_CONFIG"
echo "Seeds: $SEEDS"
echo "Wandb project base: $WANDB_PROJECT_BASE"
echo "Variant: $VARIANT"
echo "Kinetix directory: $KINETIX_DIR"
echo "Project directory: $PROJECT_DIR"
echo "========================================"

# Create projects directory
mkdir -p "$PROJECT_DIR"

cd "$KINETIX_DIR"

# Activate venv
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
    echo "Using venv: $REPO_ROOT/.venv"
fi

# Function to run training for a specific variant
run_variant() {
    local variant_name="$1"
    local monitor_dormant="$2"
    local use_redo="$3"
    local use_trac="$4"
    local wandb_project="${WANDB_PROJECT_BASE}-${variant_name}"
    local output_dir="data/outputs_noncontinual_${variant_name}"
    
    mkdir -p "$output_dir"
    
    echo ""
    echo "########################################"
    echo "# Starting $variant_name PPO"
    echo "# monitor_dormant=$monitor_dormant, use_redo=$use_redo, use_trac=$use_trac"
    echo "########################################"
    
    local failed_envs=()
    
    for env_name in "${ENVIRONMENTS[@]}"; do
        echo ""
        echo "========================================" 
        echo "[$variant_name] Training on: $env_name"
        echo "========================================"
        
        # Convert seed index to trial index (1-based)
        local trial_idx=1
        for seed in $SEEDS; do
            echo "  Seed: $seed (trial $trial_idx)"
            
            if CUDA_VISIBLE_DEVICES=$GPU python3 experiments/ppo.py \
                --config-name=$CONFIG_NAME \
                env=$ENV_CONFIG \
                train_levels=m \
                "train_levels.train_levels_list=[\"m/${env_name}.json\"]" \
                env_size=m \
                learning.monitor_dormant=$monitor_dormant \
                learning.use_redo=$use_redo \
                learning.use_trac=$use_trac \
                misc.wandb_project="$wandb_project" \
                seed=$seed \
                misc.project_dir="$PROJECT_DIR" \
                misc.trial_idx=$trial_idx \
                >> "${output_dir}/${env_name}_seed${seed}.txt" 2>&1; then
                echo "    Done (see ${output_dir}/${env_name}_seed${seed}.txt)"
            else
                echo "    WARNING: $env_name seed=$seed FAILED (exit code $?). Continuing..."
                failed_envs+=("${env_name}_seed${seed}")
            fi
            ((trial_idx++))
        done
    done
    
    echo ""
    if [ ${#failed_envs[@]} -gt 0 ]; then
        echo "[$variant_name] Completed with ${#failed_envs[@]} failure(s): ${failed_envs[*]}"
    else
        echo "[$variant_name] All environments completed successfully!"
    fi
}

# Run the requested variant(s)
# Order: trac first, then vanilla, then redo
if [[ "$VARIANT" == "all" || "$VARIANT" == "trac" ]]; then
    run_variant "trac" "False" "False" "True"
fi

if [[ "$VARIANT" == "all" || "$VARIANT" == "vanilla" ]]; then
    run_variant "vanilla" "False" "False" "False"
fi

if [[ "$VARIANT" == "all" || "$VARIANT" == "redo" ]]; then
    run_variant "redo" "True" "True" "False"
fi

echo ""
echo "========================================"
echo "All training completed!"
echo "========================================"
