#!/bin/bash
# Run PPO on Gymnax envs with parameter-based continual tasks,
# sweeping over different task_interval values (updates per task).
#
# For each task_interval, num_timesteps is auto-computed so there are 10 tasks.
#   num_timesteps = task_interval * 10 * num_envs * num_steps
#
# Per-env timesteps_per_update:
#   CartPole-v1:    2048 * 20 = 40960
#   Acrobot-v1:     2048 * 50 = 102400
#   MountainCar-v0: 2048 * 50 = 102400
#
# Usage:
#   ./run_RL_gymnax_continual_gravity_updates_sweep.sh          # all envs
#   ENVS="CartPole-v1" ./run_RL_gymnax_continual_gravity_updates_sweep.sh
#   METHODS="ppo trac redo" ./run_RL_gymnax_continual_gravity_updates_sweep.sh

set -e

GPU="${GPU:-3}"
NUM_TRIALS="${NUM_TRIALS:-5}"
ENVS="${ENVS:-CartPole-v1 MountainCar-v0 Acrobot-v1}"
METHODS="${METHODS:-ppo}"
TASK_INTERVALS="${TASK_INTERVALS:-50 100 250 500 1000 2500}"
NUM_TASKS=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# Per-env timesteps_per_update = num_envs * num_steps
declare -A TPU
TPU["CartPole-v1"]=40960      # 2048 * 20
TPU["Acrobot-v1"]=102400      # 2048 * 50
TPU["MountainCar-v0"]=102400  # 2048 * 50

echo "=========================================="
echo "RL Continual Gravity - Updates-per-task Sweep"
echo "=========================================="
echo "Trials: $NUM_TRIALS"
echo "Methods: $METHODS"
echo "Envs: $ENVS"
echo "Task intervals: $TASK_INTERVALS"
echo "GPU: $GPU"
echo "=========================================="

for env in $ENVS; do
    tpu=${TPU[$env]}
    if [[ -z "$tpu" ]]; then
        echo "ERROR: Unknown env $env, no timesteps_per_update defined"
        exit 1
    fi

    for task_interval in $TASK_INTERVALS; do
        num_timesteps=$(( task_interval * NUM_TASKS * tpu ))

        for method in $METHODS; do
            for trial in $(seq 1 $NUM_TRIALS); do
                OUTPUT_DIR="projects/gymnax/rl_continual_gravity_updates_sweep/${env}/updates_${task_interval}/${method}/trial_${trial}"
                echo ""
                echo ">>> $env | $method | updates_per_task=$task_interval | Trial $trial / $NUM_TRIALS"
                echo "    num_timesteps=$num_timesteps -> $OUTPUT_DIR"
                CUDA_VISIBLE_DEVICES=$GPU python "$REPO_ROOT/source/gymnax/train_RL_gymnax_continual.py" \
                    --env "$env" \
                    --method "$method" \
                    --task_type param \
                    --trial "$trial" \
                    --gpus "$GPU" \
                    --task_interval "$task_interval" \
                    --num_timesteps "$num_timesteps" \
                    --output_dir "$OUTPUT_DIR"
            done
        done
    done
done

echo ""
echo "Updates-per-task sweep completed!"
echo "Results in projects/gymnax/rl_continual_gravity_updates_sweep/"
