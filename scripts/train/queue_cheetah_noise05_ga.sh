#!/bin/bash
# ============================================================================
# HalfCheetah observation offset 0.5, TWO sub-tasks, the GA arm (2026-09-24).
# The paper's cheetah noise two-task cell (paper/mjx/cheetah/data/noise_2task,
# Figure 2 row "Noise, 2 tasks") had every arm but the GA: runs_mjx_noise05
# holds es nes ppo trac redo cchain (10 trials, home, 2026-09-11/12), the RL
# arms reported are CLUSTER's ant-shape re-run, and the GA was only ever
# written up for CLUSTER (submit_cheetah_noise05_2task.sh) and never submitted.
#
# Same settings as every other paper cheetah GA cell (noise_10task,
# physics_2task, actions_2task, all read 2026-09-24): 320 generations, 16 a
# phase, pop 512, 3 evals, whitening on, plasticity tracked, GA sigma 0.01
# through --ne_override (settings.py carries 0.1 for the cheetah), elite
# ratio 0.1, init around the mean. Eight trials, one a card.
#
# --checkpoint_every 20 as in queue_cheetah_div_nes_rerun.sh: the shared mjx
# runner used to hang at finalise on this machine; with the whitening-stats
# fix and periodic checkpoints all ten of that script's trials finished
# (2026-09-21). The watchdog below is the same belt and braces: a trial that
# logged generation 319 and then went quiet for an hour is killed, and the
# second pass re-runs the same command, which resumes from resume.pkl and
# finalises in a fresh process.
#
#   nohup bash scripts/train/queue_cheetah_noise05_ga.sh > logs/cheetah_noise05_ga/queue.log 2>&1 &
#
# Afterwards: link paper/mjx/cheetah/data/noise_2task/continual/ga to
# runs_mjx_noise05/mjx/continual/ga and re-extract
# (plot_stability_plasticity.py --extract, plot_generalist_scores.py --extract).
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
R=projects/iclr_2027
TREE=$R/runs_mjx_noise05
TRIALS="${TRIALS:-1 2 3 4 5 6 7 8}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
LOG=logs/cheetah_noise05_ga; mkdir -p "$LOG"
NE="--obs_norm --track_plasticity --checkpoint_every 20 --num_tasks 2 --noise_range 0.5 --ne_override sigma=0.01"
STALE_MIN=60

for t in $TRIALS; do printf 'ga\tcheetah_noise\t%s\n' "$t"; done > "$LOG/jobs.tsv"

watchdog() {
    while :; do
        for d in $TREE/mjx/continual/ga/cheetah_noise/trial_*; do
            [ -f "$d/training_metrics.json" ] && continue
            [ -f "$d/train.log" ] || continue
            grep -q "gen   319" "$d/train.log" || continue
            [ -n "$(find "$d/train.log" -mmin +$STALE_MIN)" ] || continue
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
WD=
if [ "${DRY_RUN:-0}" = 0 ]; then watchdog & WD=$!; fi

for pass in 1 2; do
    echo "pass $pass $(date -Is)"
    GPUS="$GPUS" JOBLIST=$LOG/jobs.tsv PROJECT_ROOT=$TREE ENVS=cheetah_noise \
    MJX_EXTRA="$NE" LOG_DIR=$LOG DRY_RUN="${DRY_RUN:-0}" \
        bash scripts/train/launch.sh cheetah_continual ga
    [ "${DRY_RUN:-0}" != 0 ] && break
done
[ -n "$WD" ] && { pkill -P $WD 2>/dev/null; kill $WD 2>/dev/null; }
echo "done $(date -Is)"; grep -v 'exit=0' "$LOG/status.tsv" 2>/dev/null || echo "no failures"
