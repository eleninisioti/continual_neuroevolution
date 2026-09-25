#!/bin/bash
# ============================================================================
# After scripts/train/queue_iclr_deepsea_pbt.sh: the DeepSea 12 row of the
# population diversity figure (GA, ES, PBT-PPO), appendix
# population_diversity_continual.
#
#   bash scripts/analysis/finish_deepsea_pbt.sh   [WAIT=1] [NUM_TRIALS=10]
#
# 1. waits (WAIT=1) until NUM_TRIALS PBT trials have a training_metrics.json;
# 2. gives the PBT runs the gymnax column names
#    (migrate_shared_runner_columns.py, idempotent, the pbt subtree only --
#    the other probe_deepsea arms are gymnax-trainer runs and already carry
#    them);
# 3. re-extracts and redraws the figure. The runs are linked under
#    paper/diversity/data/population/deepsea (README there).
# ============================================================================
set -eu
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
P=projects/iclr_2027
CELL=$P/probe_deepsea/gymnax/continual/pbt/DeepSea12_bsuite_sigma1.0
NUM_TRIALS="${NUM_TRIALS:-10}"; WAIT="${WAIT:-1}"
done_trials() { ls "$CELL"/trial_*/training_metrics.json 2>/dev/null | wc -l; }
if [ "$WAIT" = 1 ]; then
    while [ "$(done_trials)" -lt "$NUM_TRIALS" ]; do
        echo "$(date +%T) $(done_trials)/$NUM_TRIALS PBT trials finished"; sleep 60
    done
fi
echo "=== $(done_trials) PBT trials in $CELL ==="
# The migration walks <root>/*/*/*/trial_*; a scratch root holding one link
# keeps it to the PBT cell.
T=$(mktemp -d); mkdir -p "$T/root/continual/pbt"
ln -s "$PWD/$CELL" "$T/root/continual/pbt/DeepSea12_bsuite_sigma1.0"
$PY scripts/analysis/migrate_shared_runner_columns.py "$T/root"
rm -rf "$T"
echo "=== figure ==="
$PY scripts/analysis/plot_population_diversity.py --extract
echo "done: $P/paper/visuals/final/appendix/population_diversity_continual.{pdf,png,md}"
