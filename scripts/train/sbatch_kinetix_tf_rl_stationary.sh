#!/bin/bash
# The transformer Kinetix STATIONARY block, RL arms, on the ITU HPC: 5 arms x
# 20 levels x 3 trials = 300 array tasks, at most 10 on the cluster at once.
# Same runs as queue_iclr_kinetix_transformer.sh STAGES=rl_noncont (layout,
# seeds 41 + trial, --observation entity). A finished task exits at once, so
# resubmitting the array only redoes what is missing.
#
#   sbatch scripts/train/sbatch_kinetix_tf_rl_stationary.sh
#
# Task i: arm = ARMS[i / 60], level = LEVELS[(i / 3) % 20], trial = i % 3 + 1.
#SBATCH --job-name=kx_tf_rls
#SBATCH --partition=acltr
#SBATCH --array=0-299%10
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn16,cn17
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/kinetix_tf_hpc/%x_%A_%a.out

set -u
cd "$HOME/workspace/cont_ne_playground"
ARMS=(ppo trac redo cchain pbt)
LEVELS=($(.venv/bin/python -c "from source.envs.kinetix_levels import LEVELS; print(' '.join(LEVELS))"))
i=${SLURM_ARRAY_TASK_ID}
arm=${ARMS[$((i / 60))]}
lv=${LEVELS[$(((i / 3) % 20))]}
trial=$((i % 3 + 1))
out=projects/iclr_2027/runs_kinetix_tf/kinetix/noncontinual/$arm/Kinetix_$lv/trial_$trial
if [ -f "$out/training_metrics.json" ]; then
    echo "done already: $out"; exit 0
fi
mkdir -p "$out"
echo "$(date -Is) $(hostname) $arm $lv trial $trial"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec .venv/bin/python source/studies/kinetix/cli.py --env "Kinetix-$lv" \
    --method "$arm" --observation entity --trial "$trial" \
    --seed $((41 + trial)) --output_dir "$out"
