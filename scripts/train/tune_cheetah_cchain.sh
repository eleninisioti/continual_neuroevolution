#!/bin/bash
# ============================================================================
# Stationary HalfCheetah C-CHAIN: sweep chain_target_rel_scale at the ant PPO
# shape, on the HOME server (2026-09-17).
#
# Why: at the reference's continuous-control value 0.05 the cheetah C-CHAIN
# policy collapses in the first 5% of training (entropy -15 against PPO's -3,
# approx_kl 3-5 against 0.1) and sits near 1200 while PPO reaches ~4000. The
# goal is a value at which the stationary run is on par with the other RL
# arms. Only the setting changes, not the algorithm.
#
# Screen: UPDATES (default 9600 = 20% of the 48000 budget), 2 seeds a value,
# PPO at the same budget as the reference. Output:
#   projects/iclr_2027/tune_cheetah_cchain/<tag>/mjx/noncontinual/<arm>/cheetah/trial_N
# with <tag> = rel<value> or ppo.
#
#   bash scripts/train/tune_cheetah_cchain.sh
#   GPUS="2 3 5"  VALUES="0.2 0.05 0.01 0.002 0.0005"  SEEDS="1 2"  UPDATES=9600
#   TASK_INTERVAL=2400 (the paper stationary grid; default UPDATES/4)
#   CONFIGS="tag=key=value,key=value ..."   instead of VALUES (and no PPO row):
#       any --ppo_override keys, e.g. "off=chain_target_rel_scale=0"
#
# 2026-09-17, first screen (stopped at update 300, in _stopped_screen/): every
# value from 0.2 to 0.0005 had collapsed (entropy -10 to -17, PPO -3.5 to -4.3),
# so the next screen is CONFIGS diagnostics at UPDATES=1200.
#
# Long ant-shape RL runs on this server can hang after their last update
# (memory: mjx-ppo-probe-hung-at-finalise); train.log still carries the curve
# (scripts/analysis/recover_curve_from_train_log.py).
# ============================================================================
set -u
cd "$(dirname "$0")/../.."
GPUS=(${GPUS:-2 3 5})
VALUES=(${VALUES:-0.2 0.05 0.01 0.002 0.0005})
SEEDS=(${SEEDS:-1 2})
UPDATES=${UPDATES:-9600}
OUT=projects/iclr_2027/tune_cheetah_cchain
ANT_PPO="num_envs=512 num_steps=20 num_epochs=10 learning_rate=0.0003 reward_scale=10.0"

jobs=()
for s in "${SEEDS[@]}"; do
    if [ -n "${CONFIGS:-}" ]; then
        for c in $CONFIGS; do jobs+=("${c%%=*} cchain ${c#*=} $s"); done
    else
        jobs+=("ppo ppo - $s")
        for v in "${VALUES[@]}"; do jobs+=("rel$v cchain chain_target_rel_scale=$v $s"); done
    fi
done

i=0
for job in "${jobs[@]}"; do
    read -r tag arm value seed <<< "$job"
    gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}; i=$(( i + 1 ))
    dir=$OUT/$tag/mjx/noncontinual/$arm/cheetah/trial_$seed
    [ -f "$dir/training_metrics.json" ] && { echo "skip $dir"; continue; }
    mkdir -p "$dir"
    extra=$ANT_PPO; [ "$value" != - ] && extra="$extra ${value//,/ }"
    echo "gpu $gpu: $tag $arm seed $seed -> $dir"
    nohup .venv/bin/python source/studies/mjx/cli.py --env cheetah --method "$arm" \
        --trial "$seed" --output_dir "$dir" --gpus "$gpu" \
        --num_updates "$UPDATES" --task_interval ${TASK_INTERVAL:-$(( UPDATES / 4 ))} \
        --ppo_override $extra > "$dir/launcher.log" 2>&1 &
    sleep 5
done
wait
