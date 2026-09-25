#!/bin/bash
# ============================================================================
# Run the switch-interval sweep's missing trials ON THIS BOX, from the same
# joblist cluster would have taken (cluster was down for maintenance on
# 2026-09-21):
#
#     scripts/train/cluster/joblists/freq_missing.tsv   arm env trial sigma interval
#
# That list holds only the trials the home box had neither finished nor had in
# flight when it was generated, so running it cannot start a second copy of a
# live trial. REGENERATE it first if anything has started since.
#
# ORDER: trial-major, slowest arms first within a trial (cchain, trac, redo,
# ppo, ga, nes). Two reasons. Longest-first keeps the tail short. And if the
# run is cut at a deadline, what is missing is the LAST SEEDS of every arm
# rather than some arms entirely: the figure can then be drawn over the seeds
# every arm has, which is a smaller comparison but a fair one.
#
# GPUS is a lease pool exactly as in launch.sh -- a token per slot, taken for
# a job's lifetime and returned on exit -- with one addition: the pool is a
# named FIFO, so a card that frees up later joins without a restart:
#
#     echo 6 > logs/launch_freq_local/tokens.fifo     # one more slot on GPU 6
#
# Keep GPUs 0-2 OUT of the pool while PBT runs there: two PBT jobs fill a
# 49 GB card (~24 GB each) and a gymnax job beside them dies at CUDA init
# with an OOM that reads like a wedged driver.
#
#   nohup bash scripts/train/run_freq_missing.sh > logs/freq_local.log 2>&1 &
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LIST="${LIST:-scripts/train/cluster/joblists/freq_missing.tsv}"
GPUS="${GPUS:-3 3 3 3 4 4 4 4 5 5 5 5 7 7 7 7}"
TOTAL_GENS=4000
LOG_DIR="${LOG_DIR:-logs/launch_freq_local}"
mkdir -p "$LOG_DIR"
STATUS="$LOG_DIR/status.tsv"
FIFO="$LOG_DIR/tokens.fifo"

# Trial-major, slowest arm first.
ORDERED="$LOG_DIR/order.tsv"
awk -F'\t' 'BEGIN { r["cchain"]=0; r["trac"]=1; r["redo"]=2; r["ppo"]=3; r["ga"]=4; r["nes"]=5 }
            NF >= 5 { print $3 "\t" (($1 in r) ? r[$1] : 9) "\t" $0 }' "$LIST" \
    | sort -t$'\t' -k1,1n -k2,2n | cut -f3- > "$ORDERED"
n=$(grep -c . "$ORDERED")
echo "=== $(date -Is) $n jobs from $LIST over [$GPUS] ==="

rm -f "$FIFO"; mkfifo "$FIFO"
exec 3<>"$FIFO"                          # held open read-write: writers never block
for g in $GPUS; do echo "$g" >&3; done

i=0
while IFS=$'\t' read -r arm env trial sigma interval; do
    [ -n "${arm:-}" ] || continue
    read -r gpu <&3                      # wait for a free slot
    i=$((i + 1))
    (
        num_tasks=$(( TOTAL_GENS / interval ))
        tag="freq${interval}.${arm}.${env}.trial${trial}"
        echo ">>> [$i/$n] $(date +%H:%M) gpu $gpu : $tag"
        start=$(date +%s)
        env NUM_TRIALS=5 PROJECT_ROOT="projects/iclr_2027/runs_freq/interval${interval}" \
            TRACK_DIVERSITY=1 \
            NUM_TASKS="$num_tasks" TASK_PERIOD=10 TASK_INTERVAL="$interval" \
            SIGMAS="$sigma" GPU="$gpu" TRIALS="$trial" ENVS="$env" \
            bash scripts/train/run_experiments.sh gymnax_continual "$arm" \
            > "$LOG_DIR/$tag.log" 2>&1
        code=$?
        printf '%s\texit=%d\tsecs=%d\tgpu=%s\n' "$tag" "$code" "$(( $(date +%s) - start ))" "$gpu" >> "$STATUS"
        [ "$code" = 0 ] || echo "!! FAILED $tag (see $LOG_DIR/$tag.log)"
        echo "$gpu" >&3                  # hand the slot back
    ) &
done < "$ORDERED"
wait
echo "=== $(date -Is) done ==="
grep -v 'exit=0' "$STATUS" || echo "no failures"
