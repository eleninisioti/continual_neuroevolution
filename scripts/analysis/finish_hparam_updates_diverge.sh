#!/bin/bash
# The forgetting pass over the update-count hyperparameter sweep: learning
# accuracy LA (mean_i R[i][i]) and forgetting F of the CENTROID for every
# finished minibatch / learning-rate tree, so the "M trades forgetting against
# learning accuracy" claim is measured rather than read off Cum.
#
# One pass a setting tree (the pass takes one runs root), results under
#   projects/iclr_2027/paper/<suite>/hparam_updates/results/<setting>/centroid/
# which plot_hparam_updates.py --extract reads for the LA and F rows.
#
#   bash scripts/analysis/finish_hparam_updates_diverge.sh            # what is finished now
#   WAIT=1 bash scripts/analysis/finish_hparam_updates_diverge.sh     # after the queues
#   FAMILIES="cheetah_noise cheetah_actions" ... after ship_home.sh (mjx passes
#   want an idle GPU: memory mjx-posthoc-needs-free-gpus)
#
# A tree is skipped when it has no finished trial, or when its results hold as
# many runs as it has finished trials (FORCE=1 redoes it). Same flags as the
# width sweep's finish_hparam_sigma.sh, so the two halves' LA/F agree.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=.venv/bin/python
GPU="${GPUS:-2}"
FAMILIES="${FAMILIES:-noise actions minigrid}"
if [ "${WAIT:-0}" = 1 ]; then
    while pgrep -f "queue_iclr_hparam_updates.sh" > /dev/null; do sleep 600; done
fi
settings_of() {
    case "$1" in
        noise)    echo "reported ppo_minibatches4 ppo_minibatches8 ppo_minibatches128 ppo_epochs1 ppo_epochs3 ppo_epochs30 ppo_lr0.1x ppo_lr10x" ;;
        actions)  echo "reported_actions actions_ppo_minibatches4 actions_ppo_minibatches8 actions_ppo_minibatches128" ;;
        minigrid) echo "reported_minigrid minigrid_ppo_minibatches2 minigrid_ppo_minibatches4 minigrid_ppo_minibatches64" ;;
        cheetah_noise)   echo "reported_cheetah_noise cheetah_noise_ppo_minibatches4 cheetah_noise_ppo_minibatches8 cheetah_noise_ppo_minibatches128 cheetah_noise_ppo_lr0.1x cheetah_noise_ppo_lr10x" ;;
        cheetah_actions) echo "reported_cheetah_actions cheetah_actions_ppo_minibatches4 cheetah_actions_ppo_minibatches8 cheetah_actions_ppo_minibatches128 cheetah_actions_ppo_lr0.1x cheetah_actions_ppo_lr10x" ;;
    esac
}
for fam in $FAMILIES; do
    case "$fam" in
        noise)    SUITE=gymnax;   CELLS="CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma0.1"; NT=20 ;;
        actions)  SUITE=gymnax;   CELLS="CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma1.0"; NT=20 ;;
        minigrid) SUITE=minigrid; CELLS="MiniGrid_8x8_16x16"; NT=20 ;;
        cheetah_noise)   SUITE=mjx; CELLS="cheetah_noise";  NT=20 ;;
        cheetah_actions) SUITE=mjx; CELLS="cheetah_action"; NT=2 ;;
        *) echo "unknown family '$fam'" >&2; exit 2 ;;
    esac
    for setting in $(settings_of "$fam"); do
        ROOT=projects/iclr_2027/runs_hparam/$setting/$SUITE
        RESULTS=projects/iclr_2027/paper/$SUITE/hparam_updates/results/$setting/centroid
        n=$(find -L "$ROOT/continual/ppo" -name training_metrics.json 2>/dev/null | wc -l)   # -L: the reported trees are symlinks
        [ "$n" -gt 0 ] || { echo "--- $setting: nothing finished, skipped"; continue; }
        if [ "${FORCE:-0}" != 1 ] && [ -f "$RESULTS/behavioural_divergence.npz" ]; then
            have=$($PY -c "import numpy as np,sys; print(len(np.load(sys.argv[1])['index']))" "$RESULTS/behavioural_divergence.npz" 2>/dev/null || echo 0)
            [ "$have" -ge "$n" ] && { echo "--- $setting: $have/$n runs already scored, skipped"; continue; }
        fi
        [ "$SUITE" = minigrid ] && $PY scripts/analysis/migrate_shared_runner_columns.py "$ROOT" | tail -1
        echo "=== $(date -Is) $setting: $n finished trials -> $RESULTS"
        mkdir -p "$RESULTS"
        $PY scripts/analysis/behavioural_divergence.py \
            --runs_root "$ROOT" --methods ppo --envs $CELLS \
            --num_tasks $NT --episodes 20 --num_states 2000 --gpus "$GPU" \
            --agent_source centroid --results_dir "$RESULTS" || echo "!! $setting failed"
    done
done
echo "=== $(date -Is) done"
