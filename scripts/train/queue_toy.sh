#!/bin/bash
# =============================================================================
# The toy-landscape block, on the toy runner (scripts/train/toy_sweep.py).
#
#   bash scripts/train/queue_toy.sh                  # every stage, in order
#   ONLY="wells_d400 wells_pop" bash scripts/train/queue_toy.sh
#
# smooth, rugged    every arm the paper has, plus the dns_corrected control.
# rugged_d386       section B with 384 null coordinates: the policy's size.
# wells_d{2,50,400} the centroid question: two equal wells per relevant
#                   coordinate, d - 2 coordinates that change nothing.
# wells_pop         GA and dns_gaussian at archive sizes 8 .. 512 on the tied
#                   wells: how long drift takes to pick a well.
# projections       full-dimensional snapshots for the PCA / plane figures.
#
# Arithmetic, not rollouts: CPU (JAX_PLATFORMS=cpu), so it never competes with
# a training queue for a card. Output under $ROOT/<stage>/, one log per stage
# under $ROOT/logs/, figures and table.md written into each stage directory.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
# The repo virtualenv if there is one, as in run.sh.
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python3; fi
ROOT=${ROOT:-projects/iclr_2027/runs_toy}
SEEDS=${SEEDS:-24}
mkdir -p "$ROOT/logs"

wanted() {
    [ -z "${ONLY:-}" ] && return 0
    for s in $ONLY; do [[ "$1" == "$s"* ]] && return 0; done
    return 1
}

sweep() {       # sweep <stage> <sweep args...>
    local name=$1; shift
    wanted "$name" || return 0
    echo "=== sweep $name  $(date '+%F %T')"
    $PY scripts/train/toy_sweep.py --out_dir "$ROOT/$name" \
        --num_seeds "$SEEDS" "$@" > "$ROOT/logs/$name.log" 2>&1 \
        || echo "FAILED $name -- see $ROOT/logs/$name.log"
}

project() {     # project <stage> <method> <sigma> <level> <sweep args...>
    local name=$1 method=$2 sigma=$3 level=$4; shift 4
    wanted "$name" || return 0
    echo "=== project $name $method sigma $sigma level $level"
    $PY scripts/train/toy_sweep.py --out_dir "$ROOT/$name" \
        --project "$method" "$sigma" "$level" "$@" \
        >> "$ROOT/logs/$name.project.log" 2>&1 \
        || echo "FAILED projection $name $method"
}

ALL_ARMS="ga es dns dns_gaussian dns_corrected"

sweep smooth --landscape smooth \
    --methods $ALL_ARMS
sweep rugged --landscape rugged
sweep rugged_d386 --landscape rugged --num_params 386 --sigmas 0.1 0.2 \
    --num_seeds 16
for d in 2 50 400; do
    sweep "wells_d$d" --landscape wells --num_params "$d"
done
for pop in 16 64 1024; do
    sweep "wells_pop$pop" --landscape wells --pop_size "$pop" \
        --methods ga dns_gaussian --sigmas 0.05 --levels 0 0.01
done

# ripdim: the paper's appendix grid of the ripple stage -- a
# finer k, the paper's 24 seeds, GA and ES only.
for k in 2 4 8 16 32 64 128; do
    sweep "ripdim_k${k}" --landscape rugged --rugged_dims "$k" \
        --num_params 128 --pop_size 64 --num_seeds 24 --methods ga es \
        --sigmas 0.1 0.2 0.4 --levels 0.4 0.8 1.6
done
# rippop: does a larger population move the k at which the GA stops finding
# the generalist? The GA needs its BEST child: best-of-N gains ~sqrt(2 ln N)
# standard deviations while the cost grows ~k, so the k at which it fails
# should move only logarithmically with N. Pop 64 is ripdim itself. A
# larger pop is also a larger budget per generation: not compute-matched.
for k in 4 8 16 32 64; do
    for pop in 16 256 1024; do
        sweep "rippop_k${k}_n${pop}" --landscape rugged --rugged_dims "$k" \
            --num_params 128 --pop_size "$pop" --num_seeds 24 --methods ga es \
            --sigmas 0.2 0.4 --levels 0.4 1.6
    done
done
# rippop_matched: the same budget per sub-task as pop 64 (6400 evaluations):
# pop 256 switching every 25 generations for 250.
sweep rippop_matched_k16_n256 --landscape rugged --rugged_dims 16 \
    --num_params 128 --pop_size 256 --task_interval 25 --num_generations 250 \
    --num_seeds 24 --methods ga es --sigmas 0.2 0.4 --levels 0.4 1.6
# stiff_smooth: the smooth toy in d = 128 with c of the 126 extra coordinates
# stiff (the solutions of both sub-tasks lie within a tube of width 0.3 there),
# the rest null: does holding the shared region get harder with the
# number of directions the solutions are narrow in, rather than with d?
for c in 0 4 16 64 126; do
    sweep "stiff_smooth_c${c}" --landscape manifold --manifold_base smooth \
        --stiff_dims "$c" --tube_width 0.3 --num_params 128 --pop_size 64 \
        --num_seeds 24 --methods ga es --sigmas 0.02 0.05 0.1 0.2 \
        --levels 0.1 0.3 1.0
done
# d128: toy_local_optima in d = 128. smooth: 126 coordinates that
# change nothing (the landscape has no ripple to extend); rugged: the ripple on
# all 128. <name>_d128 sweeps the 2-D figure's sigma grid, so the sweep's own
# best sigma (and the path it saves) is chosen the way the 2-D figure's was;
# <name>_d128_wide adds sigma 0.3 / 0.4 for the table.
# plot_toy_local_optima.py --dims 128 reads both.
D128="--num_params 128 --num_seeds 24 --methods ga es"
sweep smooth_d128 --landscape smooth $D128
sweep rugged_d128 --landscape rugged --rugged_dims 128 $D128
sweep smooth_d128_wide --landscape smooth $D128 --sigmas 0.3 0.4
sweep rugged_d128_wide --landscape rugged --rugged_dims 128 $D128 --sigmas 0.3 0.4
# sigma: does the width of the basin ES settles in depend on its sigma?
# ES and the GA at seven widths on the smooth toy (no local optima), the rugged toy (local optima of one width) and the spikes toy (a
# narrow tall specialist on each shoulder of the wide shared hill), with the
# full centroid saved at every record. plot_toy_sigma_basin.py measures the
# landscape around the end-of-phase centroid and reads all three.
SIGMA="--num_seeds 24 --methods ga es --record_centroid --sigmas 0.01 0.02 0.05 0.1 0.2 0.4 0.8"
sweep sigma_smooth --landscape smooth $SIGMA
sweep sigma_rugged --landscape rugged $SIGMA --levels 0 0.4 0.8 1.6
sweep sigma_spikes --landscape spikes $SIGMA
# Many equal peaks: k relevant coordinates, each with two wells, no null ones.
# The score is the MEAN over coordinates, so one coordinate's basin choice is
# worth 1/k of it: selection on each choice weakens as k grows, and a split
# coordinate puts the centroid in its valley. Does the centroid gap grow with
# k, and does a height gap of 0.01 / 0.1 still resolve it at large k?
for k in 2 8 32 128; do
    sweep "wells_k$k" --landscape wells --relevant_dims "$k" --num_params "$k" \
        --methods ga es --sigmas 0.05 0.2 --levels 0 0.01 0.1 --num_seeds 16
done

project wells_d400 ga 0.05 0 --landscape wells --num_params 400
project wells_d400 ga 0.05 0.05 --landscape wells --num_params 400
project wells_d400 dns_gaussian 0.05 0 --landscape wells --num_params 400
project wells_d50 ga 0.05 0 --landscape wells --num_params 50

for dir in "$ROOT"/*/; do
    name=$(basename "$dir")
    [ "$name" = logs ] && continue
    wanted "$name" || continue
    [ -f "$dir/results.json" ] || continue
    echo "=== figures $name"
    $PY scripts/plotting/make_toy_figures.py "$dir" >> "$ROOT/logs/$name.log" 2>&1 \
        || echo "FAILED figures $name"
done
echo "=== done $(date '+%F %T')"
