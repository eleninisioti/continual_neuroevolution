#!/bin/bash
# Last step of the PPO hyperparameter appendix: the MiniGrid minibatch column.
# Waits for the MiniGrid launchers in flight, retries any trial that died of a
# CUDA error (run_condition re-runs only unfinished trials), then scores
# accuracy/forgetting and redraws via finish_hparam_updates.sh.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Wait on TRAINER processes only: a bare pattern on `ps -eo args` also
# matches any interactive shell whose command line quotes it, and blocked
# this loop forever on 2026-09-21.
while ps -eo args | grep -E '^[^ ]*python[^ ]* .*--output_dir [^ ]*minigrid_ppo_minibatches' -q; do sleep 120; done
echo "=== $(date -Is) MiniGrid launchers done; retrying failures"
GPUS_ALL="1 1 2 2" SETTINGS="minigrid_ppo_minibatches2 minigrid_ppo_minibatches4 minigrid_ppo_minibatches64" \
    bash scripts/train/queue_iclr_hparam_updates.sh
echo "=== $(date -Is) retry done; scoring and redrawing"
bash scripts/analysis/finish_hparam_updates.sh
