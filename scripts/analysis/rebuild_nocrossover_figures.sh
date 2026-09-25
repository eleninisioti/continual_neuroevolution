#!/bin/bash
# Re-extract and redraw the final figures whose panels changed when the GA's
# MountainCar and Kinetix runs were relinked to the no-crossover arm
# (2026-09-23, scripts/train/queue_nocrossover.sh). Everything here reads the
# paper trees through paper/<suite>/data, which finish_iclr.sh has already
# rebuilt.
#
# NOT here, because they need their own GPU compute passes or belong to
# another session: basin_width_methods / basin_width_evolution,
# landscape_slices, shared_basin, and the Appendix E switch-interval figures
# (whose GA runs still have crossover).
#
#   nohup bash scripts/analysis/rebuild_nocrossover_figures.sh > logs/nocrossover_figs.log 2>&1 &
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=.venv/bin/python
run() { echo "=== $(date -Is) $*"; "$@" || echo "!!! FAILED: $*"; }

for s in plot_noncontinual_solve plot_continual_lineplots plot_stability_plasticity \
         plot_population_diversity plot_generalist_scores plot_metrics_appendix; do
    run $PY scripts/analysis/$s.py --extract
    run $PY scripts/analysis/$s.py
done
run $PY scripts/analysis/plot_plasticity_overview.py --set main --extract
run $PY scripts/analysis/plot_plasticity_overview.py --set main
# Derived from the two figures' saved data, so it comes after them.
run $PY scripts/analysis/plot_continual_combined.py
run $PY scripts/analysis/plot_continual_combined.py --part tradeoff
run $PY scripts/analysis/plot_continual_combined.py --part curves
# The appendix table that aggregates across figures.
[ -f scripts/analysis/aggregate_significance.py ] && \
    run $PY scripts/analysis/aggregate_significance.py
echo "=== $(date -Is) figures done"
