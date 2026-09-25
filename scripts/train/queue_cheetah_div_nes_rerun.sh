#!/bin/bash
# ============================================================================
# HalfCheetah NES diversity trials, second attempt (2026-09-21). The first
# (queue_cheetah_div_nes.sh) hung at finalise (noise 1-6, actions 1-4) or died
# in the 2026-09-20 GPU wedge after generation 319 (noise 7-9, resume.pkl at
# generation 300). population_diversity_continual has no ES row for
# HalfCheetah noise and only trials 5, 8 for action reversal.
#
#   noise    runs_div_noise05_t10  nes  trials 1-10 (7-9 resume from gen 300)
#   actions  runs_div_action       nes  trials 1-4, 6, 7, 9, 10
#
# Same flags as the first attempt, plus --checkpoint_every 20: a trial that
# hangs at finalise is killed by the watchdog below and re-run with the same
# command, which resumes from resume.pkl and finalises in a fresh process.
#
# Waits until NEED_GPUS cards are idle (< 100 MiB), then runs three NE jobs a
# card on them.
#
#   nohup bash scripts/train/queue_cheetah_div_nes_rerun.sh > logs/cheetah_div_nes_rerun/queue.log 2>&1 &
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
NEED_GPUS="${NEED_GPUS:-3}"
PER_GPU="${PER_GPU:-3}"
R=projects/iclr_2027
LOG=logs/cheetah_div_nes_rerun; mkdir -p "$LOG"
NE="--obs_norm --track_plasticity --checkpoint_every 20"
STALE_MIN=60      # finalise is seconds; an hour with no log line after gen 319 is a hang

idle_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F', ' '$2 < 100 {print $1}'
}
while :; do
    free=($(idle_gpus))
    [ "${#free[@]}" -ge "$NEED_GPUS" ] && break
    sleep 120
done
free=("${free[@]:0:$NEED_GPUS}")
echo "GPUs [${free[*]}] idle at $(date -Is)"

printf 'nes\tcheetah_noise\t%s\n' 1 2 3 4 5 6 7 8 9 10 > "$LOG/noise_jobs.tsv"
printf 'nes\tcheetah_action\t%s\n' 1 2 3 4 6 7 9 10 > "$LOG/actions_jobs.tsv"
slots=""; for g in "${free[@]}"; do for _ in $(seq "$PER_GPU"); do slots="$slots $g"; done; done

# Kill a trial that logged its last generation and then went quiet: the
# launcher moves on, and the second pass below resumes it.
watchdog() {
    while :; do
        for d in $R/runs_div_noise05_t10/mjx/continual/nes/cheetah_noise/trial_* \
                 $R/runs_div_action/mjx/continual/nes/cheetah_action/trial_*; do
            [ -f "$d/training_metrics.json" ] && continue
            [ -f "$d/train.log" ] || continue
            grep -q "gen   319" "$d/train.log" || continue
            [ -n "$(find "$d/train.log" -mmin +$STALE_MIN)" ] || continue
            # the trainer is the python process writing this train.log
            log=$(readlink -f "$d/train.log")
            for pid in $(for f in /proc/[0-9]*/fd/*; do
                             [ "$(readlink "$f" 2>/dev/null)" = "$log" ] && echo "${f#/proc/}"
                         done | cut -d/ -f1 | sort -u |
                         while read -r p; do grep -q python "/proc/$p/comm" 2>/dev/null && echo "$p"; done); do
                echo "watchdog: $d hung after gen 319, killing $pid $(date -Is)"
                kill "$pid"
            done
        done
        sleep 600
    done
}
watchdog & WD=$!

for pass in 1 2; do
    echo "pass $pass $(date -Is)"
    GPUS="$slots" JOBLIST=$LOG/noise_jobs.tsv PROJECT_ROOT=$R/runs_div_noise05_t10 \
    MJX_EXTRA="$NE --num_tasks 10 --noise_range 0.5" LOG_DIR=$LOG/noise \
        bash scripts/train/launch.sh cheetah_continual nes
    GPUS="$slots" JOBLIST=$LOG/actions_jobs.tsv PROJECT_ROOT=$R/runs_div_action \
    MJX_EXTRA="$NE --num_tasks 2" LOG_DIR=$LOG/actions \
        bash scripts/train/launch.sh cheetah_continual nes
done
kill $WD 2>/dev/null
echo "done $(date -Is)"
for d in $R/runs_div_noise05_t10/mjx/continual/nes/cheetah_noise/trial_* \
         $R/runs_div_action/mjx/continual/nes/cheetah_action/trial_*; do
    [ -f "$d/training_metrics.json" ] && echo "ok      $d" || echo "MISSING $d"
done
