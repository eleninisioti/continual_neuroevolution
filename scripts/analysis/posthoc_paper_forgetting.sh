#!/usr/bin/env bash
# The centroid forgetting pass (behavioural_divergence.py, F and BD) that
# plot_stability_plasticity.py reads, for a paper data root whose reported arms
# finish_iclr.sh's own pass does not cover:
#
#   gymnax/data/physics_2task   the family pass ran on runs_param_2task, where
#                               es_arm kept OpenES (the paper's ES is NES) and
#                               where PBT does not exist: the paper tree links
#                               PBT per cell from CLUSTER (2026-09-17)
#
# One process an arm, one card each, then a merge, as the width pass does.
# The pbt shard also holds the pbt2 runs (`--methods` matches the config's
# method, `pbt` for both), so the merge leaves the pbt2 shard out; the reader
# (plot_stability_plasticity.load_pass) files a run by its directory.
#
#   GPUS="0 1 2 3 4 5 6 7" scripts/analysis/posthoc_paper_forgetting.sh
#
# -> paper/<root>/results/centroid/behavioural_divergence.{npz,json,md}
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
PAPER=projects/iclr_2027/paper
read -ra GPU_LIST <<< "${GPUS:-0}"

# root | cells | arms
ROOTS=(
    "gymnax/data/physics_2task|CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma1.0|ga nes ppo trac redo cchain pbt pbt2"
)

run_arm() {   # root cells arm gpu
    local out=$PAPER/$1/results
    mkdir -p "$out/logs"
    env XLA_PYTHON_CLIENT_PREALLOCATE=false \
        $PY scripts/analysis/behavioural_divergence.py \
        --runs_root "$PAPER/$1" --methods "$3" --envs $2 \
        --num_tasks 20 --episodes 20 --num_states 2000 --gpus "$4" \
        --agent_source centroid --results_dir "$out/centroid/shards/bd_$3" \
        > "$out/logs/bd_$3.log" 2>&1
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
    $PY scripts/analysis/behavioural_divergence.py --merge \
        "$out"/centroid/shards/bd_{cchain,ga,nes,pbt,ppo,redo,trac}/ --results_dir "$out/centroid"
done
