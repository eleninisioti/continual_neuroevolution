#!/bin/bash
# ============================================================================
# Everything that turns projects/iclr_2027/runs_param_centroid -- the PHYSICS
# sub-task tree, where a sub-task rescales the body rather than offsetting the
# observation -- into its figure and its full metrics table. The counterpart of
# finish_iclr_centroid.sh, which does the same for the observation-noise tree;
# the two differ only in the root, the results directory and the output names.
#
# `--sigma 1.0` below is how the analysis scripts recover the environment from
# a cell name, NOT a setting of these runs: no observation offset is drawn in
# this tree; see the header of scripts/train/queue_iclr_param_centroid.sh, which
# is the queue that fills it. Run once that sweep has finished. The
# noncontinual phase is a SYMLINK into runs_centroid -- sub-task 0 of a param
# run is the stock body -- so it is already complete.
#
#   1. verify   both phases are complete, compute-matched and boundary-aligned
#               (CLAUDE.md rule (c)). Nothing downstream runs if this fails --
#               a figure drawn over a tree that lost a trial is worse than no
#               figure, because it looks fine.
#   2. evaluate the post-hoc pass that fills the ZT column: every sub-task's
#               saved agent, re-rolled on its own sub-task and on the NEXT one
#               before any search has touched it.
#   3. diverge  the pass that fills F and BD: forgetting, and the fraction of
#               a sub-task agent's visited states on which the next sub-task's
#               agent acts differently.
#   4. dormancy the per-unit dormant masks at one checkpoint per sub-task,
#               which is what tells a DEAD unit from a merely sparse one --
#               the dormant fraction alone cannot.
#   5. plastic  the plasticity figure: dormancy, churn, weight RMS, NTK rank
#               and step, plus the persistence panel from step 4.
#   6. plot     the lineplot, legend and table, at BOTH reported metrics.
#
# `elite_eval` and `centroid` are two different questions and the paper wants
# both spelled out:
#   elite_eval  what the best individual scores, re-scored out of sample.
#               The honest version of the old `best_fitness` curve.
#   centroid    what the mean of the population's WEIGHTS scores. For ES/NES
#               that IS the incumbent; for GA and DNS it measures whether the
#               archive has collapsed, and is NOT their performance.
# ============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY=.venv/bin/python
# THE ROOT IS runs_param_centroid, NOT runs_param. Every NE trial in
# runs_param was started before the trainers began writing the centroid on
# 2026-09-09, so `--metric centroid` and the centroid plasticity figure would
# silently describe the ELITE there -- checked: no `centroid` key in any
# runs_param checkpoints.npz. runs_param_centroid is the re-cut,
# scripts/train/queue_iclr_param_centroid.sh.
ROOT="${ROOT:-projects/iclr_2027/runs_param_centroid/gymnax}"
RESULTS="${RESULTS:-projects/iclr_2027/results/gymnax_param/sigma1.0}"
FIGS=projects/iclr_2027/figures
# The eight REPORTED arms. The Iso+LineDD column (`ga_isoline`, `dns`) is the
# operator ablation; the paper reports the gaussian column only, so `ga`
# (gaussian+truncation) and `dns_gaussian` (gaussian+novelty) are the NE pair
# here, exactly as in finish_iclr_centroid.sh.
ARMS="${ARMS:-ga dns_gaussian es nes ppo trac redo cchain}"
mkdir -p "$RESULTS" "$FIGS"

echo "=== 1/6 verify ==="
$PY scripts/verify_runs.py "$ROOT" --phase noncontinual
$PY scripts/verify_runs.py "$ROOT" --phase continual

echo "=== 2/6 post-hoc evaluation (ZT) ==="
$PY -m source.studies.evaluate_continual \
    --root "$ROOT/continual" --episodes 100 --gpus 0

echo "=== 3/6 behavioural divergence (F, BD) ==="
$PY scripts/analysis/behavioural_divergence.py \
    --runs_root "$ROOT" --methods $ARMS \
    --envs CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma1.0 \
    --num_tasks 20 --episodes 20 --num_states 2000 \
    --results_dir "$RESULTS"

echo "=== 4/6 per-unit dormancy checkpoints (persistence, step) ==="
# JAX_PLATFORMS=cpu: this walks every checkpoint of every trial and the nets
# are 386 parameters, so the GPU buys nothing and the training queue wants the
# cards. CUDA_VISIBLE_DEVICES="" is NOT enough -- jax still tries the cuda
# backend and dies on "No visible GPU devices".
JAX_PLATFORMS=cpu $PY scripts/analysis/plasticity_checkpoints.py \
    --runs_root "$ROOT" --phase continual --sigma 1.0 \
    --out "$RESULTS"

echo "=== 5/6 plasticity figure ==="
# Goes AFTER the lineplot in the paper: that says which methods keep learning
# across a switch, this says which of the mechanisms the plasticity literature
# blames are actually present. The persistence panel is the one that separates
# dead units from functionally sparse ones, and it needs the pass above.
JAX_PLATFORMS=cpu $PY scripts/make_plasticity_figure.py "$ROOT" \
    --phase continual --sigma 1.0 \
    --checkpoints "$RESULTS" --agent "${agent:-elite}" \
    --out "$FIGS/param_plasticity_sigma1.0"

echo "=== 6/6 figures and tables ==="
for metric in elite_eval centroid; do
    $PY scripts/make_lineplot.py "$ROOT" \
        --phase continual --sigma 1.0 --metric "$metric" \
        --results-dir "$RESULTS" \
        --out "$FIGS/param_continual_${metric}_sigma1.0"
done
$PY scripts/make_lineplot.py "$ROOT" \
    --phase noncontinual --metric elite_eval \
    --out "$FIGS/param_noncontinual_elite_eval"

echo "=== done ==="
