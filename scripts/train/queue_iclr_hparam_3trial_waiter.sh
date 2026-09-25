#!/bin/bash
# Re-queues the action-reversal and MiniGrid sigma sweeps at THREE trials
# (the user's call, 2026-09-20 09:20) once the trials their first launch
# (five trials) left in flight have finished; trial_4/5 were killed and
# removed, trials 1-3 in flight are kept and skipped by run_condition.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
while ps -eo args | grep -q '[t]rain_.*runs_hparam_actions\|[c]li.py.*runs_hparam_minigrid'; do sleep 120; done
echo "=== $(date -Is) in-flight trials done; re-queuing at 3 trials"
FAMILY=minigrid NUM_TRIALS=3 bash scripts/train/queue_iclr_hparam.sh > logs/queue_iclr_hparam_minigrid.log 2>&1 &
FAMILY=actions NUM_TRIALS=3 GPUS_A="3 3 4" GPUS_B="4 5 5" GPUS_C="6 6 7 7" bash scripts/train/queue_iclr_hparam.sh > logs/queue_iclr_hparam_actions.log 2>&1 &
wait
