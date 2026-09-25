#!/usr/bin/env bash
# The centroid width pass (curvature_width.py) that plot_basin_width_methods.py
# reads, for the data roots whose runs no finish_iclr.sh pass describes:
#
#   gymnax/data/physics_2task      finish_iclr.sh's pass (2026-09-15) predates the
#                                  Acrobot link-mass x1.15 swap and has no PBT
#   mjx/cheetah/data/noise_2task   offset 0.5, CLUSTER ant-shape RL; never scored
#
# ABS=1 runs the absolute-noise pass (--width-abs-only) instead, on every tanh
# root of the figure (HalfCheetah, Kinetix): those panels report absolute width,
# since weight scale is not a symmetry of a tanh network.
#
#   GPUS="4 6 7" scripts/analysis/posthoc_paper_width.sh
#   GPUS="0 1 2" ABS=1 scripts/analysis/posthoc_paper_width.sh
#
# -> paper/<root>/results/centroid/curvature_width.json
#    paper/<root>/results/centroid_abswidth/curvature_width.json   (ABS=1)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
PAPER=projects/iclr_2027/paper
read -ra GPU_LIST <<< "${GPUS:-0}"

# root | cells | arms
if [ "${ABS:-0}" = 1 ]; then
    OUT=centroid_abswidth; PREFIX=aw; EXTRA=--width-abs-only
    ROOTS=(
        "mjx/cheetah/data/noise_2task|cheetah_noise|nes ppo trac redo cchain pbt pbt2"
        "mjx/cheetah/data/physics_2task|cheetah_friction|ga es nes ppo trac redo cchain"
        "mjx/cheetah/data/actions_2task|cheetah_action|ga es nes ppo trac redo cchain pbt"
        "mjx/cheetah/data/noise_10task|cheetah_noise|ga es nes ppo trac redo cchain pbt pbt2"
        "kinetix/data|Kinetix20|ga es nes ppo trac redo cchain pbt pbt2"
    )
else
    OUT=centroid; PREFIX=cw; EXTRA=
    ROOTS=(
        "gymnax/data/physics_2task|CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma1.0|ga nes ppo trac redo cchain pbt pbt2"
        "mjx/cheetah/data/noise_2task|cheetah_noise|nes ppo trac redo cchain pbt pbt2"
    )
fi

run_arm() {   # root cells arm gpu
    local out=$PAPER/$1/results
    mkdir -p "$out/logs"
    env CUDA_VISIBLE_DEVICES=$4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        $PY scripts/analysis/curvature_width.py "$PAPER/$1" --cells $2 \
        --agent centroid --methods "$3" $EXTRA --out "$out/$OUT/shards/${PREFIX}_$3" \
        > "$out/logs/${PREFIX}_$3.log" 2>&1
    echo "done $1 $3"
}

jobs=()
for spec in "${ROOTS[@]}"; do
    IFS='|' read -r root cells arms <<< "$spec"
    for arm in $arms; do jobs+=("$root|$cells|$arm"); done
done
[ "${MERGE_ONLY:-0}" = 1 ] && jobs=()
i=0
for job in "${jobs[@]+"${jobs[@]}"}"; do
    IFS='|' read -r root cells arm <<< "$job"
    run_arm "$root" "$cells" "$arm" "${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}" &
    i=$((i + 1))
    if (( i % ${#GPU_LIST[@]} == 0 )); then wait; fi
done
wait

for spec in "${ROOTS[@]}"; do
    out=$PAPER/${spec%%|*}/results
    $PY scripts/analysis/curvature_width.py --merge \
        "$out"/$OUT/shards/${PREFIX}_*/curvature_width.json --out "$out/$OUT"
done
