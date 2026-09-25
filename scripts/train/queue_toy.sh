#!/bin/bash
# =============================================================================
# The toy-landscape block, on the shared runner (source/studies/toy/sweep.py).
#
#   bash scripts/train/queue_toy.sh                  # every stage, in order
#   ONLY="wells_d400 wells_pop" bash scripts/train/queue_toy.sh
#
# smooth, rugged    sections A and B of docs/generalists/generalist_report.html,
#                   re-run with every arm the paper has (ga_isoline and
#                   dns_gaussian were never on them) plus the toy controls.
#                   smooth also carries section A's optimizer x shaping arms.
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
PY=.venv/bin/python
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
    $PY -m source.studies.toy.sweep --out_dir "$ROOT/$name" \
        --num_seeds "$SEEDS" "$@" > "$ROOT/logs/$name.log" 2>&1 \
        || echo "FAILED $name -- see $ROOT/logs/$name.log"
}

project() {     # project <stage> <method> <sigma> <level> <sweep args...>
    local name=$1 method=$2 sigma=$3 level=$4; shift 4
    wanted "$name" || return 0
    echo "=== project $name $method sigma $sigma level $level"
    $PY -m source.studies.toy.sweep --out_dir "$ROOT/$name" \
        --project "$method" "$sigma" "$level" "$@" \
        >> "$ROOT/logs/$name.project.log" 2>&1 \
        || echo "FAILED projection $name $method"
}

ALL_ARMS="ga ga_isoline dns dns_gaussian es nes dns_corrected ga_stale nes_mu nes_mu_select"

sweep smooth --landscape smooth \
    --methods $ALL_ARMS nes_adam es_sgd es_nomom nes_adam_nomom
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

# The manifold grid. Does a large enough population find the generalist
# through the ripple, does its centroid then track its elite, and does the
# population's spread count the directions that change nothing? d = 100 with
# c stiff coordinates (98 - c free ones), four population sizes, sigma swept
# against a tube width of 0.1. ga_isoline breeds at the reference's corrected
# line_sigma 0.05 here, since at 0.5 it diverges and says nothing about this.
for c in 0 16 48 96; do
    for pop in 16 64 256 1024; do
        sweep "manifold_c${c}_n${pop}" --landscape manifold --num_params 100 \
            --stiff_dims "$c" --pop_size "$pop" --num_seeds 16 \
            --methods ga ga_isoline dns_gaussian nes --line_sigma 0.05 \
            --sigmas 0.02 0.05 0.1 0.2
    done
done

# What a GA needs to find AND keep a generalist (2026-09-14).
# trade  E1: specialists at 0.65 (below the 0.70 generalist, the rugged
#        default) .. 0.85 (above it), switching every 25 / 100 / 400
#        generations, 2 or 100 coordinates. Kept only if nothing better on
#        the current sub-task is reachable within one interval?
# tube   E2: the generalist inside a tube of width w in c coordinates. Found
#        only if some sigma crosses the ripple (~0.2) and still fits the tube
#        (sigma sqrt(c) < w)?
# ripple E4: the ripple along k coordinates, same barrier per coordinate. Do
#        local optima in more dimensions stop the GA?
# drift  E3: the smooth toy, where the GA finds the generalist and loses it by
#        drift on each sub-task's plateau. Does ga_focus_fine, which pulls the
#        population together, keep it?
# ga_focus_fine is on every stage: does centroid tracking help or hurt?
for sp in 0.65 0.70 0.75 0.85; do
    for T in 25 100 400; do
        for d in 2 100; do
            sweep "trade_s${sp}_T${T}_d${d}" --landscape rugged \
                --specialist_peak "$sp" --task_interval "$T" \
                --num_generations 1600 --num_params "$d" --num_seeds 16 \
                --methods ga ga_focus_fine nes --sigmas 0.05 0.2 \
                --levels 0 0.4 1.6
        done
    done
done
for w in 0.1 0.3 1.0; do
    for c in 4 16 64; do
        for pop in 64 256; do
            sweep "tube_w${w}_c${c}_n${pop}" --landscape manifold \
                --num_params 100 --stiff_dims "$c" --tube_width "$w" \
                --pop_size "$pop" --num_seeds 16 --methods ga ga_focus_fine \
                --sigmas 0.02 0.05 0.1 0.2 0.4 --levels 0 0.8 1.6
        done
    done
done
for k in 2 8 32 128; do
    for pop in 64 256; do
        sweep "ripple_k${k}_n${pop}" --landscape rugged --rugged_dims "$k" \
            --num_params 128 --pop_size "$pop" --num_seeds 16 \
            --methods ga ga_focus_fine nes --sigmas 0.02 0.05 0.1 0.2 0.4 \
            --levels 0.4 0.8 1.6
    done
done
for d in 2 50; do
    for T in 25 100 400; do
        sweep "drift_d${d}_T${T}" --landscape smooth --num_params "$d" \
            --task_interval "$T" --num_generations 1600 --num_seeds 16 \
            --methods ga ga_focus_fine nes --sigmas 0.02 0.05 0.1 0.2
    done
done
# ripdim: the paper's appendix grid of the ripple stage (2026-09-17) -- a
# finer k, the paper's 24 seeds, GA and ES (NES) only.
# plot_toy_ripple_dims.py reads it.
for k in 2 4 8 16 32 64 128; do
    sweep "ripdim_k${k}" --landscape rugged --rugged_dims "$k" \
        --num_params 128 --pop_size 64 --num_seeds 24 --methods ga nes \
        --sigmas 0.1 0.2 0.4 --levels 0.4 0.8 1.6
done
# rippop: does a larger population move the k at which the GA stops finding
# the generalist (2026-09-18)? The GA needs its BEST child: best-of-N gains
# ~sqrt(2 ln N) standard deviations while the cost grows ~k, so the k at which it
# fails should move only logarithmically with N. Pop 64 is ripdim itself. A
# larger pop is also a larger budget per generation: not compute-matched.
for k in 4 8 16 32 64; do
    for pop in 16 256 1024; do
        sweep "rippop_k${k}_n${pop}" --landscape rugged --rugged_dims "$k" \
            --num_params 128 --pop_size "$pop" --num_seeds 24 --methods ga nes \
            --sigmas 0.2 0.4 --levels 0.4 1.6
    done
done
# rippop_matched: the same budget per sub-task as pop 64 (6400 evaluations):
# pop 256 switching every 25 generations for 250.
sweep rippop_matched_k16_n256 --landscape rugged --rugged_dims 16 \
    --num_params 128 --pop_size 256 --task_interval 25 --num_generations 250 \
    --num_seeds 24 --methods ga nes --sigmas 0.2 0.4 --levels 0.4 1.6
# stiff_smooth: the smooth toy in d = 128 with c of the 126 extra coordinates
# stiff (the solutions of both sub-tasks lie within a tube of width 0.3 there),
# the rest null (2026-09-22): does holding the shared region get harder with
# the number of directions the solutions are narrow in, rather than with d?
for c in 0 4 16 64 126; do
    sweep "stiff_smooth_c${c}" --landscape manifold --manifold_base smooth \
        --stiff_dims "$c" --tube_width 0.3 --num_params 128 --pop_size 64 \
        --num_seeds 24 --methods ga nes --sigmas 0.02 0.05 0.1 0.2 \
        --levels 0.1 0.3 1.0
done
# d128: toy_local_optima in d = 128 (2026-09-18). smooth: 126 coordinates that
# change nothing (the landscape has no ripple to extend); rugged: the ripple on
# all 128. <name>_d128 sweeps the 2-D figure's sigma grid, so the sweep's own
# best sigma (and the path it saves) is chosen the way the 2-D figure's was;
# <name>_d128_wide adds sigma 0.3 / 0.4 for the table.
# plot_toy_local_optima.py --dims 128 reads both.
D128="--num_params 128 --num_seeds 24 --methods ga nes"
sweep smooth_d128 --landscape smooth $D128
sweep rugged_d128 --landscape rugged --rugged_dims 128 $D128
sweep smooth_d128_wide --landscape smooth $D128 --sigmas 0.3 0.4
sweep rugged_d128_wide --landscape rugged --rugged_dims 128 $D128 --sigmas 0.3 0.4
# sigma: does the width of the basin ES settles in depend on its sigma
# (2026-09-18)? ES (NES) and the GA at seven widths on the smooth toy (no local
# optima), the rugged toy (local optima of one width) and the spikes toy (a
# narrow tall specialist on each shoulder of the wide shared hill), with the
# full centroid saved at every record. plot_toy_sigma_basin.py measures the
# landscape around the end-of-phase centroid and reads all three.
SIGMA="--num_seeds 24 --methods ga nes --record_centroid --sigmas 0.01 0.02 0.05 0.1 0.2 0.4 0.8"
sweep sigma_smooth --landscape smooth $SIGMA
sweep sigma_rugged --landscape rugged $SIGMA --levels 0 0.4 0.8 1.6
sweep sigma_spikes --landscape spikes $SIGMA
# ga_keep: a child must strictly beat a member to replace it. Does freezing
# the archive on a plateau keep the smooth generalist (drift), and what does
# it cost where finding needs moves between equal scores (trade, ripple)?
for d in 2 50; do
    for T in 25 100 400; do
        sweep "keep_drift_d${d}_T${T}" --landscape smooth --num_params "$d" \
            --task_interval "$T" --num_generations 1600 --num_seeds 16 \
            --methods ga_keep --sigmas 0.02 0.05 0.1 0.2
    done
done
for T in 25 100 400; do
    sweep "keep_trade_s0.85_T${T}_d100" --landscape rugged \
        --specialist_peak 0.85 --task_interval "$T" --num_generations 1600 \
        --num_params 100 --num_seeds 16 --methods ga_keep --sigmas 0.05 0.2 \
        --levels 0 0.4 1.6
done
for k in 2 8 32 128; do
    sweep "keep_ripple_k${k}_n64" --landscape rugged --rugged_dims "$k" \
        --num_params 128 --pop_size 64 --num_seeds 16 --methods ga_keep \
        --sigmas 0.02 0.05 0.1 0.2 0.4 --levels 0.4 0.8 1.6
done

# Many equal peaks: k relevant coordinates, each with two wells, no null ones.
# The score is the MEAN over coordinates, so one coordinate's basin choice is
# worth 1/k of it: selection on each choice weakens as k grows, and a split
# coordinate puts the centroid in its valley. Does the centroid gap grow with
# k, and does a height gap of 0.01 / 0.1 still resolve it at large k?
for k in 2 8 32 128; do
    sweep "wells_k$k" --landscape wells --relevant_dims "$k" --num_params "$k" \
        --methods ga nes --sigmas 0.05 0.2 --levels 0 0.01 0.1 --num_seeds 16
done

# ga_track against ga: the gaussian GA whose sigma shrinks while its archive's
# centroid scores below the median elite and grows back (never past the
# starting sigma) once it does not (source/studies/generalists/ne.py). The
# swept sigma is the starting width. NES is the centroid-is-the-search control.
TRACK_ARMS="ga ga_track ga_success ga_survive ga_subspace ga_merge_noise ga_merge_pc nes"
# ga_merge_track: rank noise and the sigma shrink from one centroid signal, at
# the two sigma rates the min-wells toy separated (0.3 was the only setting to
# track by generation 200 at both k = 2 and k = 8). Safety on smooth and
# rugged, and the two regimes it has to fix: equal additive wells and
# epistatic wells.
for rate in 0.1 0.3; do
    COMBO="--methods ga ga_merge_track nes --merge_rate 0.1 --merge_max 2.0 --sigma_rate $rate --num_seeds 16"
    sweep "combo_s${rate}_rugged" --landscape rugged $COMBO
    sweep "combo_s${rate}_smooth" --landscape smooth $COMBO
    sweep "combo_s${rate}_wells_d50" --landscape wells --num_params 50 \
        --sigmas 0.2 0.5 --levels 0 0.05 $COMBO
    sweep "combo_s${rate}_wells_min_k8" --landscape wells --combine min \
        --relevant_dims 8 --num_params 8 --sigmas 0.05 0.2 --levels 0 0.01 $COMBO
done

# Epistatic (weakest-coordinate) wells: the candidate Kinetix-like toy, where
# a child that moves any coordinate off its well is ruined and a centroid in
# the valley of any split coordinate fails.
for k in 2 8 32; do
    sweep "wells_min_k$k" --landscape wells --combine min --relevant_dims "$k" \
        --num_params "$k" --methods ga ga_track ga_merge_noise ga_merge_pc \
        ga_subspace nes --sigmas 0.05 0.2 --levels 0 0.01 --num_seeds 16
done
sweep track_wells_d50 --landscape wells --num_params 50 --methods $TRACK_ARMS \
    --sigmas 0.2 0.5 --levels 0 0.05 --num_seeds 16
sweep track_manifold_c48 --landscape manifold --num_params 100 \
    --stiff_dims 48 --pop_size 256 --methods $TRACK_ARMS --sigmas 0.05 0.2 \
    --levels 0 1.6 --num_seeds 16
sweep track_rugged --landscape rugged --methods $TRACK_ARMS --num_seeds 16
sweep track_smooth --landscape smooth --methods $TRACK_ARMS --num_seeds 16

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
    $PY scripts/make_toy_figures.py "$dir" >> "$ROOT/logs/$name.log" 2>&1 \
        || echo "FAILED figures $name"
done
echo "=== done $(date '+%F %T')"
