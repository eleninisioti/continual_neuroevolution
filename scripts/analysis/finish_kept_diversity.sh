#!/bin/bash
# ============================================================================
# The `kept` diversity figure: forgetting passes for every kept cell, then the
# figure (scripts/analysis/plot_diversity_metrics.py --figure kept).
#
#   scripts/analysis/finish_kept_diversity.sh [GPUS=0] [FORCE=0] [PASSES=1]
#
# The kept cells and where their runs come from are the `kept_*` roots under
# projects/iclr_2027/paper/diversity/data (see its README). The forgetting
# passes (behavioural_divergence.py, ELITE = finalgen) go to
# paper/diversity/results/kept_<fam>/elite/<tree>_<arm>/, one directory per
# (tree, arm), which is what plot_diversity_metrics.kept_divergence reads;
# CartPole/Acrobot GA and ES come from the paper pass and are not re-run.
# A pass is skipped when its behavioural_divergence.json exists (FORCE=1 redoes
# it), so this is safe to re-run as cells complete.
#
# DeepSea episodes are 12 steps, so --episodes is raised there to give the
# divergence probe enough states (the pass warns below 2000 states per
# sub-task otherwise).
# ============================================================================
set -eu
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
GPUS="${GPUS:-0}"; FORCE="${FORCE:-0}"; PASSES="${PASSES:-1}"
P=projects/iclr_2027
RES=$P/paper/diversity/results

pass() {   # pass <kept root> <runs tree (under projects/iclr_2027)> <arm> <episodes> <cells...>
    local root=$1 tree=$2 arm=$3 episodes=$4; shift 4
    local out=$RES/$root/elite/$(basename "$tree" | tr '/' '_')_$arm
    if [ "$FORCE" = 0 ] && [ -f "$out/behavioural_divergence.json" ]; then
        echo "  have $out"; return
    fi
    mkdir -p "$out"
    $PY scripts/analysis/behavioural_divergence.py \
        --runs_root "$P/$tree/gymnax/continual" --methods "$arm" --envs "$@" \
        --num_tasks 20 --episodes "$episodes" --num_states 2000 --gpus "$GPUS" \
        --agent_source finalgen --results_dir "$out"
}

if [ "$PASSES" = 1 ]; then
    echo "=== forgetting passes (elite) ==="
    # noise: the paper pass covers CartPole/Acrobot GA and ES in runs_centroid
    pass kept_noise runs_centroid_dnstier dns_gaussian 20 CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0
    for arm in ga nes dns_gaussian; do
        pass kept_noise runs_centroid $arm 20 MountainCar_v0_sigma0.5
    done
    # action reversal: the paper pass covers GA/ES (MountainCar GA from _ga_plain_backup)
    pass kept_actions runs_actions_dnstier dns_gaussian 20 \
        CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma1.0
    # action map (DeepSea 12): everything is new
    for arm in ga nes dns_gaussian; do
        pass kept_deepsea probe_deepsea $arm 200 DeepSea12_bsuite_sigma1.0
    done
fi

echo "=== figure ==="
$PY scripts/analysis/plot_diversity_metrics.py --figure kept --extract
echo "done: $P/paper/visuals/final/diversity_metrics_kept_elite.{pdf,png,md}"
