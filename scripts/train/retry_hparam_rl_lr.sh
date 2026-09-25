#!/bin/bash
# Retries the rl_lr0.1x and MiniGrid-minibatch trials that died of CUDA
# errors (driver instability, 2026-09-20/21), once their first launch is over.
# run_condition skips any trial whose training_metrics.json exists, so this
# re-runs exactly the failures. It waits for (a) the rl_lr0.1x queue to have
# moved on to the ES settings and (b) no MiniGrid sweep trainer running, so a
# retried trial is never started twice.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
until grep -q 'nes_pop2048 ->' logs/queue_hparam_rl_lr.log 2>/dev/null \
      && ! ps -eo args | grep -q '[m]inigrid_ppo_minibatches'; do
    sleep 300
done
echo "=== $(date -Is) first launches over; retrying failures"
GPUS_ALL="1 1 2 2 1 1" SETTINGS="rl_lr0.1x minigrid_ppo_minibatches2 minigrid_ppo_minibatches4 minigrid_ppo_minibatches64" \
    bash scripts/train/queue_iclr_hparam_updates.sh
echo "=== $(date -Is) retry done"
