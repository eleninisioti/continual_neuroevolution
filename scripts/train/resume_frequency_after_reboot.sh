#!/bin/bash
# ============================================================================
# Resume the switch-frequency appendix sweep after the 2026-09-20 reboot.
#
# The box's GPU driver wedged around 16:50 on 2026-09-20 with interval 50 at
# 94 of 120 trials and interval 400 not yet started; the jobs in flight at
# that moment are listed in logs/wedge_20260920_inflight.txt. Nothing is lost
# but those trials: run_condition skips any trial whose training_metrics.json
# exists, so this picks up exactly where the queue stopped.
#
#   nohup bash scripts/train/resume_frequency_after_reboot.sh \
#       > logs/queue_frequency_resume.log 2>&1 &
#
# TWO PASSES, because `pbt2` ran out of GPU memory on 14 of 15 trials at four
# jobs a GPU (the same arm is fine in runs_centroid, so this is contention,
# not the arm):
#
#   1. the pbt arms alone at TWO jobs a GPU
#   2. everything else at four, the setting the 94 finished trials ran at
#
# GPUs 0-2 only, as before: 3-7 are left for the peer sessions' trees, which
# also lost their in-flight trials to the wedge.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "=== $(date -Is) pass 1/2: pbt arms, 2 jobs a GPU ==="
GPUS="0 0 1 1 2 2" ARMS="pbt pbt2" \
    bash scripts/train/queue_iclr_frequency.sh

echo "=== $(date -Is) pass 2/2: the rest, 4 jobs a GPU ==="
GPUS="0 0 0 0 1 1 1 1 2 2 2 2" ARMS="ga nes ppo trac redo cchain" \
    bash scripts/train/queue_iclr_frequency.sh

echo "=== $(date -Is) done ==="
