#!/bin/bash
# Completes the MiniGrid column of the PPO hyperparameter appendix and
# refreshes its figures.
#
#   nohup setsid bash scripts/analysis/finish_hparam_updates.sh > logs/finish_hparam_updates.log 2>&1 &
#
# Waits for the MiniGrid minibatch settings only -- NOT for the whole queue,
# whose ES settings no longer feed this section (the ES subsection was cut on
# 2026-09-20; Appendix D.3, the width sweep, carries ES). Then, in order:
# migrates the shared-runner columns, scores learning accuracy and forgetting,
# redraws, and copies the three figures the paper uses into the Overleaf
# clone. Committing and pushing them is left to a human.
#
# Safe to re-run: every step skips work already done.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OVERLEAF="${OVERLEAF:-$HOME/workspace/iclr_overleaf}"
SETTINGS="minigrid_ppo_minibatches2 minigrid_ppo_minibatches4 minigrid_ppo_minibatches64"

# Done when every setting has its trials, or when nothing is training any more
# (a setting whose jobs died is reported rather than waited on forever).
while :; do
    pending=0
    for s in $SETTINGS; do
        n=$(find "projects/iclr_2027/runs_hparam/$s" -name training_metrics.json 2>/dev/null | wc -l)
        [ "$n" -ge "${NUM_TRIALS:-5}" ] || pending=$((pending + 1))
    done
    [ "$pending" -eq 0 ] && break
    # `[m]inigrid` so this loop cannot match its own command line.
    ps -eo args | grep -q '[m]inigrid_ppo_minibatches' || {
        echo "=== $(date -Is) $pending setting(s) unfinished and nothing training; going ahead"
        break
    }
    sleep 600
done

for s in $SETTINGS; do
    printf '%-32s %s trials\n' "$s" \
        "$(find "projects/iclr_2027/runs_hparam/$s" -name training_metrics.json 2>/dev/null | wc -l)"
done
grep -hv 'exit=0' logs/launch_hparam_minigrid_*/status.tsv 2>/dev/null || echo "failures: none"

GPUS="${GPUS:-4}" FAMILIES=minigrid bash scripts/analysis/finish_hparam_updates_diverge.sh || exit 1
.venv/bin/python scripts/analysis/plot_hparam_updates.py --extract || exit 1

# Only the three the paper includes; the lr figures are drawn but unused.
mkdir -p "$OVERLEAF/images/appendix"
cp projects/iclr_2027/paper/visuals/final/appendix/hparam_{sweep,minibatches,minibatches_plane}.pdf \
   "$OVERLEAF/images/appendix/"
echo "=== $(date -Is) figures refreshed in $OVERLEAF/images/appendix/ -- commit and push them yourself"
