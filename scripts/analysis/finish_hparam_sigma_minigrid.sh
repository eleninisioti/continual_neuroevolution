#!/bin/bash
# ============================================================================
# The MiniGrid row of Appendix D's search-width figure, added after the
# four-hour version (noise + action reversal) shipped on 2026-09-21.
#
#     nohup bash scripts/analysis/finish_hparam_sigma_minigrid.sh > logs/finish_hparam_sigma_minigrid.log 2>&1 &
#
# Waits for all 24 MiniGrid trials (8 widths x 3), scores them with the
# centroid forgetting pass and redraws the figure with all three families.
# It waits on the trial COUNT, not on queue processes: finish_hparam_sigma.sh's
# wait pattern also matches queue_iclr_hparam_updates.sh, the update-count
# half's launcher, which outlives its session and would hold it for ever.
# Copies the figure into the Overleaf clone; the MiniGrid paragraph and the
# caption's third row are written by hand.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
R=projects/iclr_2027/runs_hparam_minigrid/minigrid/continual
OVERLEAF="${OVERLEAF:-$HOME/workspace/iclr_overleaf}"
count() { ls $R/*_sigma*/*/trial_*/training_metrics.json 2>/dev/null | wc -l; }
echo "=== $(date -Is) waiting for 24 MiniGrid trials ($(count) now)"
while [ "$(count)" -lt 24 ]; do sleep 600; done
echo "=== $(date -Is) all in; scoring MiniGrid"
WAIT=0 FAMILIES=minigrid STEPS=diverge GPUS="${GPUS:-4}" \
    bash scripts/analysis/finish_hparam_sigma.sh || exit 1
.venv/bin/python scripts/analysis/plot_hparam_sigma.py --extract || exit 1
cp projects/iclr_2027/paper/visuals/final/appendix/hparam_sigma.pdf \
   projects/iclr_2027/paper/visuals/final/appendix/hparam_sigma_curves.pdf \
   "$OVERLEAF/images/appendix/" 2>/dev/null && echo "=== copied to Overleaf clone (not committed)"
echo "=== $(date -Is) done"
