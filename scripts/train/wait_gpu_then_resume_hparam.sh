#!/bin/bash
# ============================================================================
# Start resume_hparam_sigma.sh as soon as the box has room for it.
#
#     nohup bash scripts/train/wait_gpu_then_resume_hparam.sh > logs/wait_gpu_hparam.log 2>&1 &
#
# WHY A WAITER. After the 2026-09-20 reboot every card is committed: the
# switch-frequency sweep holds GPUs 0-5 and a DeepSea PBT sweep holds 6 and 7,
# and a gymnax PBT job takes ~23 GB, two a card. At 20:23 the search-width
# sweep was queued onto 6 and 7 anyway and 23 trials died on
# CUDA_ERROR_OUT_OF_MEMORY before a context could even be created: those two
# cards have a few hundred MiB free, less than a CUDA context. Nothing is
# wrong with the queue; there is no memory. (Those 23 wrote no
# training_metrics.json, so they are re-run, not lost.)
#
# So: poll, and hand the freed cards to the queue in the order they free.
# NEED_MB (default 4000) is what a card must have free to count as usable --
# four gymnax NE jobs at ~600 MiB plus the context and some headroom. A card
# is taken only if it has been free for two consecutive polls, so a gap
# between two of another sweep's jobs is not mistaken for the end of it.
#
# WATCH (default "6 7") is which cards may be taken. It is NOT every card
# with room: GPUs 0-5 carry the switch-frequency sweep for ~12 hours from
# 2026-09-20 20:11, and three of those cards are nearly empty in MEMORY while
# being busy in COMPUTE, so free memory alone would walk straight into another
# session's sweep. 6 and 7 are the two that session offered.
#
# POLL (default 300 s), NEED_MB, MIN_GPUS (default 1): how many usable cards
# to wait for before starting. The slot lists are built from the cards found,
# four jobs a card, and passed to resume_hparam_sigma.sh, which then runs the
# noise, action-reversal and MiniGrid families in that order.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

WATCH="${WATCH:-6 7}"
POLL="${POLL:-300}"
NEED_MB="${NEED_MB:-4000}"
MIN_GPUS="${MIN_GPUS:-1}"
PER_GPU="${PER_GPU:-4}"

free_gpus() {
    # index of every card with at least NEED_MB free, one per line. A hung
    # driver makes nvidia-smi block for ever, so it is bounded: a timeout
    # prints nothing, which reads as "no card is free" and simply waits.
    timeout 60 nvidia-smi --query-gpu=index,memory.total,memory.used \
        --format=csv,noheader,nounits 2>/dev/null |
    while IFS=, read -r i total used; do
        i="${i// /}"
        case " $WATCH " in *" $i "*) ;; *) continue ;; esac
        [ $(( ${total// /} - ${used// /} )) -ge "$NEED_MB" ] && echo "$i"
    done
}

echo "=== $(date -Is) waiting for $MIN_GPUS of GPUs [$WATCH] to have ${NEED_MB} MiB free (poll ${POLL}s)"
prev=""
while :; do
    now=$(free_gpus | tr '\n' ' ')
    # The intersection with the previous poll: free twice in a row.
    stable=""
    for g in $now; do case " $prev " in *" $g "*) stable="$stable $g" ;; esac; done
    n=$(echo $stable | wc -w)
    echo "$(date -Is) free now [$now] stable [$stable]"
    if [ "$n" -ge "$MIN_GPUS" ]; then
        break
    fi
    prev="$now"
    sleep "$POLL"
done

# Slot lists: PER_GPU jobs a card, dealt round robin over the three launchers
# a gymnax family runs (they must be disjoint -- launch.sh leases per
# launcher). With one card: A and B get two slots each, C one.
set -- $stable
A=""; B=""; C=""
i=0
for g in "$@"; do
    for _ in $(seq 1 "$PER_GPU"); do
        case $((i % 3)) in 0) A="$A $g" ;; 1) B="$B $g" ;; *) C="$C $g" ;; esac
        i=$((i + 1))
    done
done
# No launcher may get an empty list: launch.sh reads that as "every visible GPU".
[ -n "$A" ] || A="$1"; [ -n "$B" ] || B="$1"; [ -n "$C" ] || C="$1"

echo "=== $(date -Is) starting on [$stable]: A=[$A] B=[$B] C=[$C]"
GPUS_NOISE_A="$A" GPUS_NOISE_B="$B" GPUS_NOISE_C="$C" \
GPUS_ACTIONS_A="$A" GPUS_ACTIONS_B="$B" GPUS_ACTIONS_C="$C" \
GPUS_MINIGRID="$(echo $stable | sed 's/\([0-9]*\)/\1 \1/g')" \
    bash scripts/train/resume_hparam_sigma.sh
