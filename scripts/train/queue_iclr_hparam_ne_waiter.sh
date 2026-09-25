#!/bin/bash
# Re-queues the NE half of queue_iclr_hparam.sh once the trials in flight
# from its first launch (2026-09-19 20:05, killed at the launcher level so the
# ppo_lr/ppo_ent arms would not be queued -- the PPO sweep is the other
# session's) have finished: a trial without training_metrics.json is not
# skipped by run_condition, so relaunching while one is still training would
# start it twice.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
while ps -eo args | grep -q '[t]rain_.*runs_hparam/gymnax/continual'; do sleep 120; done
echo "=== $(date -Is) in-flight trials done; queuing the NE arms"
NE_ONLY=1 bash scripts/train/queue_iclr_hparam.sh
