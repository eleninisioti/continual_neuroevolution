#!/bin/bash
# ============================================================================
# Everything that turns projects/iclr_2027/runs_centroid into the paper's two
# gymnax figures and their tables. Run once the training queues have finished.
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
#   5. plastic  the plasticity figure: dormancy, persistence, action shift,
#               weight RMS, NTK rank and step, plus the persistence panel.
#   6. plot     the lineplot, legend and table, at BOTH reported metrics.
#
# ONE GRID, THREE SIGMAS. The reported cells use a DIFFERENT observation noise
# per environment -- sigma 1.0 is 7-15x MountainCar's velocity range, so that
# column was noise rather than a perturbation:
#
#     CartPole-v1      sigma 1.0
#     Acrobot-v1       sigma 1.0
#     MountainCar-v0   sigma 0.1     <- NOT 1.0
#
# `--sigma` is one suffix for a whole figure and cannot express that, which is
# why there used to be a second script, finish_iclr_centroid_mcar_sigma0.1.sh,
# drawing MountainCar on its own. Two scripts meant two arm lists, two env
# lists and two sigmas that all had to agree, and the three panels came out as
# two files that cannot be read as one result. `--cells` names the directories
# instead, `CELLS` below is the single place the grid is stated, and every pass
# takes the same list -- so the lineplot, the table, the checkpoint pass and the
# plasticity figure cannot describe different cells.
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
ROOT=projects/iclr_2027/runs_centroid/gymnax
# One directory per REPORTED NETWORK, mirrored between results/ and figures/:
# `elite` is the performance number, `centroid` the collapse diagnostic. Both
# come from the same trials over the same budget -- see figures/README.md.
RESULTS=projects/iclr_2027/results/gymnax_centroid
FIGS=projects/iclr_2027/figures

# THE REPORTED GRID. Stated once, passed to every pass below.
CELLS="${CELLS:-CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma0.1}"
# THE REPORTED ARMS. `ga` is the gaussian-mutation GA and `dns_gaussian` the
# gaussian novelty arm, so the novelty comparison differs in the SELECTION RULE
# alone. The Iso+LineDD pair (`ga_isoline`, `dns`) is out as of 2026-09-09:
# Iso+LineDD compounds population variance -- `ga_isoline` saturated float32 on
# 10/10 MountainCar trials -- so a mixed comparison tested the operator and the
# selection rule at once.
#
# Drop from this list any arm with no finished trial in ANY cell, or step 6
# exits naming it. As of 2026-09-09 that is `es` and `nes`.
ARMS="${ARMS:-ga dns_gaussian es nes ppo trac redo cchain}"
mkdir -p "$RESULTS"/{elite,centroid}/target "$FIGS"/{elite,centroid}

echo "=== grid: $CELLS"
echo "=== arms: $ARMS"

echo "=== 1/6 verify ==="
$PY scripts/verify_runs.py "$ROOT" --phase noncontinual
$PY scripts/verify_runs.py "$ROOT" --phase continual

echo "=== 2/6 post-hoc evaluation (ZT) ==="
# Tree-wide and idempotent: it skips a run whose ZT columns are already filled
# unless --force, so pointing it at the whole continual root costs nothing.
$PY -m source.studies.evaluate_continual \
    --root "$ROOT/continual" --episodes 100 --gpus 0

echo "=== 3/6 behavioural divergence (F, BD) ==="
$PY scripts/analysis/behavioural_divergence.py \
    --runs_root "$ROOT" --methods $ARMS \
    --envs $CELLS \
    --num_tasks 20 --episodes 20 --num_states 2000 \
    --results_dir "$RESULTS/elite/target"

echo "=== 4/6 per-unit dormancy checkpoints (persistence, step) ==="
# JAX_PLATFORMS=cpu: this walks every checkpoint of every trial and the nets
# are 386 parameters, so the GPU buys nothing and the training queue wants the
# cards. CUDA_VISIBLE_DEVICES="" is NOT enough -- jax still tries the cuda
# backend and dies on "No visible GPU devices".
#
# Once per reported network. `--agent` picks `finalgen` or `centroid` out of
# checkpoints.npz, and is passed to the FIGURE too, so the curve rows read
# `ne_centroid_*` under centroid instead of describing the elite in both
# figures the way they used to. Runs written before the trainers logged those
# columns fall back to the elite ones and both scripts say so on stdout -- if
# you see that warning, the NE trials predate the change and need re-running
# before the centroid figure means what it says.
for agent in elite centroid; do
    JAX_PLATFORMS=cpu $PY scripts/analysis/plasticity_checkpoints.py \
        --runs_root "$ROOT" --phase continual --cells $CELLS \
        --agent "$agent" --methods $ARMS \
        --out "$RESULTS/$agent/target"
done

echo "=== 5/6 plasticity figure ==="
# Goes AFTER the lineplot in the paper: that says which methods keep learning
# across a switch, this says which of the mechanisms the plasticity literature
# blames are actually present. The persistence panel is the one that separates
# dead units from functionally sparse ones, and it needs the pass above.
for agent in elite centroid; do
    JAX_PLATFORMS=cpu $PY scripts/make_plasticity_figure.py "$ROOT" \
        --phase continual --cells $CELLS \
        --checkpoints "$RESULTS/$agent/target" --agent "$agent" --methods $ARMS \
        --out "$FIGS/$agent/plasticity"
done

echo "=== 6/6 figures and tables ==="
# `elite_eval` -> figures/elite/, `centroid` -> figures/centroid/. F and BD
# are a property of the saved sub-task agents, not of the plotted curve, so
# both tables read the one `elite` behavioural_divergence.json.
for pair in "elite_eval elite" "centroid centroid"; do
    set -- $pair
    $PY scripts/make_lineplot.py "$ROOT" \
        --phase continual --cells $CELLS --metric "$1" --methods $ARMS \
        --results-dir "$RESULTS/elite/target" \
        --out "$FIGS/$2/continual"
done
# The stationary reference, elite only: `centroid` has no separate one to be.
# No --cells: the noncontinual cells carry no sigma in their names.
$PY scripts/make_lineplot.py "$ROOT" \
    --phase noncontinual --metric elite_eval --methods $ARMS \
    --out "$FIGS/elite/noncontinual"

echo "=== done ==="
