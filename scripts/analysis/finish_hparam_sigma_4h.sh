#!/bin/bash
# ============================================================================
# The four-hour version of Appendix D.3: observation noise and action
# reversal only (the user's call, 2026-09-21 07:40). MiniGrid is left out
# because one MiniGrid ES trial takes ~370 minutes even on a GH200, so it
# cannot land inside four hours whatever cards it gets; it joins later
# through finish_hparam_sigma.sh.
#
#     nohup bash scripts/analysis/finish_hparam_sigma_4h.sh > logs/finish_hparam_sigma_4h.log 2>&1 &
#
# Waits for every action-reversal trial of the sweep (8 arms x 3 cells x 3
# trials) to have its training_metrics.json, scores that family with the
# centroid forgetting pass (the noise family is already scored), draws the
# figure with --families noise actions, and copies it into the Overleaf clone
# for a human to commit. GPUS (default 6) is the pass's card.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=.venv/bin/python
R=projects/iclr_2027/runs_hparam_actions/gymnax/continual
OVERLEAF="${OVERLEAF:-$HOME/workspace/iclr_overleaf}"
NEED=72     # 8 widths x 3 cells x 3 trials

count() { ls $R/{ga,nes}_sigma*/*/trial_*/training_metrics.json 2>/dev/null | wc -l; }

echo "=== $(date -Is) waiting for $NEED action-reversal trials ($(count) now)"
while [ "$(count)" -lt "$NEED" ]; do sleep 300; done
echo "=== $(date -Is) all $NEED in; scoring the action-reversal family"

WAIT=0 FAMILIES=actions STEPS=diverge GPUS="${GPUS:-6}" \
    bash scripts/analysis/finish_hparam_sigma.sh || exit 1

$PY scripts/analysis/plot_hparam_sigma.py --extract --families noise actions || exit 1
if [ -d "$OVERLEAF/images/appendix" ]; then
    cp projects/iclr_2027/paper/visuals/final/appendix/hparam_sigma.pdf \
       projects/iclr_2027/paper/visuals/final/appendix/hparam_sigma_curves.pdf \
       "$OVERLEAF/images/appendix/"
    echo "=== figures copied to $OVERLEAF/images/appendix/ (not committed)"
fi
echo "=== $(date -Is) done"
