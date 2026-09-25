#!/bin/bash
# ============================================================================
# Resume the three search-width sweeps after the 2026-09-20 GPU wedge
# (driver hung ~16:50, box rebooted 20:11; logs/wedge_20260920_inflight.txt
# lists what was running). A trial without training_metrics.json is re-run
# from scratch by run_condition, so this only has to be started, not fixed up.
#
#     nohup bash scripts/train/resume_hparam_sigma.sh > logs/resume_hparam_sigma.log 2>&1 &
#
# ONE FAMILY AT A TIME, AND ONLY FOUR JOBS. The switch-frequency sweep holds
# GPUs 0-5 and a DeepSea PBT sweep holds 6 and 7 (2026-09-20 20:30): a gymnax
# PBT job takes ~23 GB, two a card, so 6 and 7 have about 2 GB free each and
# the gymnax NE jobs this queues are ~600 MiB. Two a card is what fits; a
# third would OOM whenever the PBT jobs peak. Raise the slot lists as soon as
# either sweep ends. The order is the appendix's: the noise cells are the
# figure's first row and are 80/120 done, so they finish first.
#
# GPUS_NOISE_A/B/C, GPUS_ACTIONS_A/B/C and GPUS_MINIGRID override the slot
# lists (the launchers inside one family must be DISJOINT -- launch.sh leases
# per launcher; an EMPTY list means launch.sh picks every visible GPU, so
# never leave one unset). When a card frees up, kill this and restart it with
# wider lists; everything finished by then is skipped.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "=== $(date -Is) resuming the search-width sweeps on GPUs 6 and 7, two jobs a card ==="

# --- noise: 5 seeds, NE arms only (the PPO half is the other session's) -----
GPUS_A="${GPUS_NOISE_A:-6 6}" GPUS_B="${GPUS_NOISE_B:-7 7}" \
GPUS_C="${GPUS_NOISE_C:-6 7}" NE_ONLY=1 NUM_TRIALS=5 \
    bash scripts/train/queue_iclr_hparam.sh
echo "=== $(date -Is) noise done ==="

# --- action reversal: 3 seeds ----------------------------------------------
FAMILY=actions NUM_TRIALS=3 \
GPUS_A="${GPUS_ACTIONS_A:-6 6}" GPUS_B="${GPUS_ACTIONS_B:-7 7}" \
GPUS_C="${GPUS_ACTIONS_C:-6 7}" \
    bash scripts/train/queue_iclr_hparam.sh
echo "=== $(date -Is) action reversal done ==="

# --- MiniGrid: 3 seeds, one launcher. Fewer slots a card: the shared runner's
# MiniGrid jobs are a CNN over a 16x16 grid, not a 16x16 MLP.
FAMILY=minigrid NUM_TRIALS=3 GPUS_A="${GPUS_MINIGRID:-6 7}" \
    bash scripts/train/queue_iclr_hparam.sh
echo "=== $(date -Is) MiniGrid done ==="

echo "=== $(date -Is) all three families queued out; failures:"
grep -hv 'exit=0' logs/launch_hparam_{CartPole,Acrobot,MountainCar}/status.tsv \
    logs/launch_hparam_actions_*/status.tsv logs/launch_hparam_minigrid/status.tsv 2>/dev/null \
    || echo "  none"
