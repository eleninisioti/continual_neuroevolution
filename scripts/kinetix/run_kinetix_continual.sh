#!/bin/bash

# Continual PPO training on 20 medium Kinetix tasks (h0-h19)
# Runs sequentially, using checkpoint from previous task
# Runs 3 variants: Vanilla PPO, Redo PPO, TRAC PPO
# Uses pixel-based observations by default
#
# Usage:
#   ./run_kinetix_continual.sh                   # Run vanilla on GPU 0
#   ./run_kinetix_continual.sh --cuda 3          # Run on GPU 3
#   ./run_kinetix_continual.sh --symbolic        # Run with symbolic observations
#   ./run_kinetix_continual.sh --variant vanilla # Run only vanilla PPO
#   ./run_kinetix_continual.sh --variant redo    # Run only redo PPO
#   ./run_kinetix_continual.sh --variant trac    # Run only trac PPO
#   ./run_kinetix_continual.sh --variant all     # Run all 3 variants sequentially
#   ./run_kinetix_continual.sh --num_trials 10   # Run 10 trials per variant

# NOTE: We do NOT use 'set -e' here so that a failure on one
# environment does not kill the entire script (and skip remaining envs/variants).

# Default settings
GPU=3
CONFIG_NAME="ppo"
ENV_CONFIG="pixels"  # Default to pixels
NUM_TRIALS=1
WANDB_PROJECT_BASE="Kinetix-continual"
VARIANT="vanilla"  # vanilla, redo, trac, or all

# 20 medium h-tasks (in order for continual learning)
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
        --num_trials)
            NUM_TRIALS="$2"
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
            echo "Usage: $0 [--cuda GPU] [--symbolic] [--num_trials N] [--project PROJECT] [--variant vanilla|redo|trac|all]"
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
echo "Kinetix Continual Training"
echo "========================================"
echo "GPU: $GPU"
echo "Observation type: $ENV_CONFIG"
echo "Num trials: $NUM_TRIALS"
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

# Function to extract checkpoint path from wandb log
extract_checkpoint_from_log() {
    local log_file="$1"
    if [ -f "$log_file" ]; then
        # Look for the checkpoint path in the log
        # The log contains: "Parameters of model saved in checkpoints/kinetix/RUN_NAME-TIMESTAMP-TIMESTEPS/full_model.pbz2"
        grep "Parameters of model saved in" "$log_file" | tail -1 | sed 's/.*saved in //' | sed 's/\/full_model\.pbz2.*//'
    fi
}

# Function to convert local path to wandb artifact path
convert_to_wandb_path() {
    local local_path="$1"
    local wandb_project="$2"
    
    # Extract the run name from the local path
    # Format: checkpoints/kinetix/RUN_NAME-TIMESTAMP-TIMESTEPS/
    local run_name=$(basename "$local_path")
    
    # Convert to wandb artifact format
    echo "eleni/${wandb_project}/${run_name}-checkpoint:v0"
}

# Function to run continual training for a specific variant
run_continual_variant() {
    local variant_name="$1"
    local monitor_dormant="$2"
    local use_redo="$3"
    local use_trac="$4"
    local wandb_project="${WANDB_PROJECT_BASE}-${variant_name}"
    local output_dir="data/outputs_continual_${variant_name}"
    
    mkdir -p "$output_dir"
    
    echo ""
    echo "########################################"
    echo "# Starting $variant_name PPO (Continual)"
    echo "# monitor_dormant=$monitor_dormant, use_redo=$use_redo, use_trac=$use_trac"
    echo "########################################"
    
    for trial_idx in $(seq 1 $NUM_TRIALS); do
        echo ""
        echo "=========================================="
        echo "Starting Trial $trial_idx (${variant_name})"
        echo "=========================================="
        
        # Use trial_idx as the seed for reproducibility
        seed=${trial_idx}
        
        # Track failures per trial
        failed_envs=()
        
        # Run the first environment without checkpoint
        echo "Running first environment: ${ENVIRONMENTS[0]}"
        
        if CUDA_VISIBLE_DEVICES=$GPU python3 experiments/ppo.py \
            --config-name=$CONFIG_NAME \
            env=$ENV_CONFIG \
            train_levels=m \
            "train_levels.train_levels_list=[\"m/${ENVIRONMENTS[0]}.json\"]" \
            env_size=m \
            +continual=True \
            learning.monitor_dormant=$monitor_dormant \
            learning.use_redo=$use_redo \
            learning.use_trac=$use_trac \
            misc.wandb_project="$wandb_project" \
            seed=$seed \
            misc.project_dir="$PROJECT_DIR" \
            misc.trial_idx=$trial_idx \
            >> "${output_dir}/trial${trial_idx}_env01_${ENVIRONMENTS[0]}.txt" 2>&1; then
            echo "First environment completed. Checking for checkpoint..."
        else
            echo "WARNING: First environment ${ENVIRONMENTS[0]} FAILED (exit code $?). Continuing..."
            failed_envs+=("${ENVIRONMENTS[0]}")
        fi
        
        # Run subsequent environments using checkpoints from previous runs
        for i in $(seq 1 $((${#ENVIRONMENTS[@]} - 1))); do
            prev_env="${ENVIRONMENTS[$((i-1))]}"
            curr_env="${ENVIRONMENTS[$i]}"
            env_num=$((i + 1))
            env_num_padded=$(printf "%02d" $env_num)
            prev_env_num_padded=$(printf "%02d" $i)
            prev_log="${output_dir}/trial${trial_idx}_env${prev_env_num_padded}_${prev_env}.txt"
            
            echo ""
            echo "Running environment ${env_num}/20: $curr_env"
            echo "Using checkpoint from: $prev_env"
            
            # Extract checkpoint path from previous run's log
            checkpoint_path=$(extract_checkpoint_from_log "$prev_log")
            
            # Build the training command
            local train_cmd="CUDA_VISIBLE_DEVICES=$GPU python3 experiments/ppo.py"
            train_cmd="$train_cmd --config-name=$CONFIG_NAME"
            train_cmd="$train_cmd env=$ENV_CONFIG"
            train_cmd="$train_cmd train_levels=m"
            train_cmd="$train_cmd \"train_levels.train_levels_list=[\\\"m/${curr_env}.json\\\"]\""
            train_cmd="$train_cmd env_size=m"
            train_cmd="$train_cmd +continual=True"
            train_cmd="$train_cmd learning.monitor_dormant=$monitor_dormant"
            train_cmd="$train_cmd learning.use_redo=$use_redo"
            train_cmd="$train_cmd learning.use_trac=$use_trac"
            train_cmd="$train_cmd misc.wandb_project=$wandb_project"
            train_cmd="$train_cmd seed=$seed"
            train_cmd="$train_cmd misc.project_dir=$PROJECT_DIR"
            train_cmd="$train_cmd misc.trial_idx=$trial_idx"
            
            if [ -n "$checkpoint_path" ]; then
                wandb_checkpoint=$(convert_to_wandb_path "$checkpoint_path" "$wandb_project")
                echo "Using checkpoint: $wandb_checkpoint"
                train_cmd="$train_cmd misc.load_from_checkpoint=$wandb_checkpoint"
            else
                echo "Warning: Could not find checkpoint from previous run. Running without checkpoint."
            fi
            
            if eval $train_cmd >> "${output_dir}/trial${trial_idx}_env${env_num_padded}_${curr_env}.txt" 2>&1; then
                echo "Completed environment ${env_num}/20: $curr_env"
            else
                echo "WARNING: Environment ${env_num}/20 $curr_env FAILED (exit code $?). Continuing..."
                failed_envs+=("$curr_env")
            fi
        done
        
        if [ ${#failed_envs[@]} -gt 0 ]; then
            echo "WARNING: Trial $trial_idx had ${#failed_envs[@]} failed env(s): ${failed_envs[*]}"
        else
            echo "Trial $trial_idx completed successfully!"
        fi
    done
    
    echo ""
    echo "=========================================="
    echo "[$variant_name] All $NUM_TRIALS trials completed!"
    echo "Total environments per trial: ${#ENVIRONMENTS[@]}"
    echo "Total experiments: $((NUM_TRIALS * ${#ENVIRONMENTS[@]}))"
    echo "=========================================="
}

# Run the requested variant(s)
# Order: redo first, then trac, then vanilla
if [[ "$VARIANT" == "all" || "$VARIANT" == "redo" ]]; then
    run_continual_variant "redo" "True" "True" "False"
fi

if [[ "$VARIANT" == "all" || "$VARIANT" == "trac" ]]; then
    run_continual_variant "trac" "False" "False" "True"
fi

if [[ "$VARIANT" == "all" || "$VARIANT" == "vanilla" ]]; then
    run_continual_variant "vanilla" "False" "False" "False"
fi

echo ""
echo "========================================"
echo "All continual training completed!"
echo "========================================"
