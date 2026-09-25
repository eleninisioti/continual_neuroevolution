#!/usr/bin/env bash
# The centroid checkpoint passes plot_plasticity_overview.py reads, for the
# cheetah runs the final paper figures report (paper/mjx/cheetah/data/<family>,
# whose RL arms and noise GA are the CLUSTER 2026-09-16 runs). finish_iclr.sh
# scored the home trees, so its results under paper/mjx/cheetah/<family>/results
# describe other RL runs. Arms whose link points at a tree finish_iclr.sh
# already scored have their shards copied from there instead of re-run.
#
#   GPUS="4 6 7" scripts/analysis/posthoc_paper_cheetah.sh
#
# -> paper/mjx/cheetah/data/{noise_10task,actions_2task}/results/
#        centroid/plasticity_checkpoints.json          weight RMS
#        centroid/curvature_width.json                 NTK rank
#        centroid_pooled/plasticity_checkpoints.json   dormant fraction, age
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
PAPER=projects/iclr_2027/paper/mjx/cheetah
read -ra GPU_LIST <<< "${GPUS:-0}"

# family  cell  arms-to-run  arms-to-copy  finish_iclr.sh results dir
FAMILIES=(
    "noise_10task cheetah_noise|ga ppo trac redo cchain pbt pbt2|nes es|$PAPER/noise/10task/results"
    "actions_2task cheetah_action|ppo trac redo cchain pbt|nes es ga|$PAPER/actions/2task/results"
)

run_arm() {   # family cell arm gpu
    local root=$PAPER/data/$1 out=$PAPER/data/$1/results
    local env=(env CUDA_VISIBLE_DEVICES=$4 XLA_PYTHON_CLIENT_PREALLOCATE=false)
    mkdir -p "$out/logs"
    {
        "${env[@]}" $PY scripts/analysis/plasticity_checkpoints.py --runs_root "$root" \
            --phase continual --cells "$2" --agent centroid --methods "$3" \
            --out "$out/centroid/shards/$3"
        "${env[@]}" $PY scripts/analysis/plasticity_checkpoints.py --runs_root "$root" \
            --phase continual --cells "$2" --agent centroid --probe pooled \
            --criterion magnitude --no-curvature --methods "$3" \
            --out "$out/centroid_pooled/shards/$3"
        "${env[@]}" $PY scripts/analysis/curvature_width.py "$root" --cells "$2" \
            --agent centroid --methods "$3" --out "$out/centroid/shards/cw_$3"
    } > "$out/logs/$3.log" 2>&1
    echo "done $1 $3"
}

jobs=()
for fam in "${FAMILIES[@]}"; do
    IFS='|' read -r head run copy old <<< "$fam"
    read -r family cell <<< "$head"
    out=$PAPER/data/$family/results
    for arm in $copy; do
        for d in centroid/shards/$arm centroid/shards/cw_$arm centroid_pooled/shards/$arm; do
            mkdir -p "$out/$(dirname "$d")"
            rm -rf "$out/$d"
            cp -r "$old/$d" "$out/$d"
        done
        # Same runs, reached through the paper's links: record the paper tree as
        # the root, which curvature_width.py --merge requires every shard to share.
        $PY - "$out/centroid/shards/cw_$arm/curvature_width.json" "$PAPER/data/$family" <<'PYEOF'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
blob = json.loads(path.read_text())
blob['meta']['root'] = sys.argv[2]
path.write_text(json.dumps(blob))
PYEOF
    done
    for arm in $run; do
        jobs+=("$family $cell $arm")
    done
done

# One job a GPU at a time, round-robin over GPU_LIST. MERGE_ONLY=1 skips the
# passes (their shards are already there) and only copies and merges.
[ "${MERGE_ONLY:-0}" = 1 ] && jobs=()
i=0
for job in "${jobs[@]+"${jobs[@]}"}"; do
    run_arm $job "${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}" &
    i=$((i + 1))
    if (( i % ${#GPU_LIST[@]} == 0 )); then wait; fi
done
wait

for fam in "${FAMILIES[@]}"; do
    IFS='|' read -r head _ _ _ <<< "$fam"
    out=$PAPER/data/${head%% *}/results
    $PY scripts/analysis/merge_plasticity_checkpoints.py \
        $(ls -d "$out"/centroid/shards/*/plasticity_checkpoints.json | grep -v '/cw_') \
        --out "$out/centroid/plasticity_checkpoints.json"
    $PY scripts/analysis/merge_plasticity_checkpoints.py \
        "$out"/centroid_pooled/shards/*/plasticity_checkpoints.json \
        --out "$out/centroid_pooled/plasticity_checkpoints.json"
    $PY scripts/analysis/curvature_width.py --merge \
        "$out"/centroid/shards/cw_*/curvature_width.json --out "$out/centroid"
done
