#!/bin/bash
# The transformer Kinetix20 chain for the RL arms that run on the ITU HPC
# (see queue_iclr_kinetix_transformer.sh for the grid; PBT-PPO and C-CHAIN run
# on the home server). One array task = one (arm, trial), one GPU.
#
#   sbatch scripts/train/sbatch_kinetix_tf_rl_chain.sh           # all 9
#   sbatch --array=4 scripts/train/sbatch_kinetix_tf_rl_chain.sh # one
#
# Task i: arm = ARMS[i / 3], trial = i % 3 + 1, seed 41 + trial -- the same
# seeds as the local chains. A resume point every 2000 updates
# (--checkpoint_every), so a resubmitted task continues rather than restarts;
# a finished one (training_metrics.json present) exits at once.
#SBATCH --job-name=kx_tf_rl
#SBATCH --partition=acltr
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn16,cn17
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/kinetix_tf_hpc/%x_%A_%a.out

set -u
cd "$HOME/workspace/cont_ne_playground"
ARMS=(ppo trac redo)
i=${SLURM_ARRAY_TASK_ID}
arm=${ARMS[$((i / 3))]}
trial=$((i % 3 + 1))
out=projects/iclr_2027/runs_kinetix_tf/kinetix/continual/$arm/Kinetix20/trial_$trial
if [ -f "$out/training_metrics.json" ]; then
    echo "done already: $out"; exit 0
fi
mkdir -p "$out"
echo "$(date -Is) $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-?} $arm trial $trial"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec .venv/bin/python source/studies/kinetix/cli.py --env Kinetix20 \
    --method "$arm" --observation entity --trial "$trial" \
    --seed $((41 + trial)) --checkpoint_every 2000 --output_dir "$out"
