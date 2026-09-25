#!/bin/bash
# ============================================================================
# The search-width half of the hyperparameter appendix: scores the sigma
# sweeps (scripts/train/queue_iclr_hparam.sh, FAMILY=noise|actions|minigrid)
# with the centroid forgetting pass and draws the stability-plasticity plane
# over sigma (plot_hparam_sigma.py, one row a family).
#
#     nohup bash scripts/analysis/finish_hparam_sigma.sh > logs/finish_hparam_sigma.log 2>&1 &
#
# Trees and passes, one a family:
#     noise     runs_hparam/gymnax           -> paper/gymnax/hparam/results/centroid
#     actions   runs_hparam_actions/gymnax   -> paper/gymnax/hparam_actions/results/centroid
#     minigrid  runs_hparam_minigrid/minigrid -> paper/minigrid/hparam/results/centroid
#
# WAIT=1 (default) first waits for every queue, waiter and trainer of the
# three sweeps to finish; the pass is idempotent (a scored run is skipped),
# so re-running after more trials land only scores the new ones. FAMILIES
# selects the sweeps, STEPS="diverge plot" the passes, GPUS the pass's GPU
# (default 3). OVERLEAF (default the clone in ~/workspace) gets the figures
# copied into images/appendix/.
#
# The other half of the appendix -- PPO epochs / rollout length / learning
# rate and ES population / width / learning rate against cumulative return --
# is plot_hparam_updates.py over runs_hparam/<setting>/, driven by
# finish_hparam_updates.sh, and is not touched here.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=.venv/bin/python
FAMILIES="${FAMILIES:-noise actions minigrid}"
STEPS="${STEPS:-diverge plot}"
OVERLEAF="${OVERLEAF:-$HOME/workspace/iclr_overleaf}"
has() { case " $STEPS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if [ "${WAIT:-1}" = 1 ]; then
    while ps -eo args | grep -q '[q]ueue_iclr_hparam\|[t]rain_.*runs_hparam/gymnax/continual\|[t]rain_.*runs_hparam_actions\|[c]li.py.*runs_hparam_minigrid'; do
        sleep 600
    done
    echo "=== $(date -Is) sweeps finished; failures:"
    grep -hv 'exit=0' logs/launch_hparam_{CartPole,Acrobot,MountainCar}/status.tsv \
        logs/launch_hparam_actions_*/status.tsv logs/launch_hparam_minigrid/status.tsv 2>/dev/null || echo "  none"
fi

if has diverge; then
    for fam in $FAMILIES; do
        case "$fam" in
            noise)    ROOT=projects/iclr_2027/runs_hparam/gymnax
                      RESULTS=projects/iclr_2027/paper/gymnax/hparam/results/centroid
                      CELLS="CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma0.1" ;;
            actions)  ROOT=projects/iclr_2027/runs_hparam_actions/gymnax
                      RESULTS=projects/iclr_2027/paper/gymnax/hparam_actions/results/centroid
                      CELLS="CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma1.0" ;;
            minigrid) ROOT=projects/iclr_2027/runs_hparam_minigrid/minigrid
                      RESULTS=projects/iclr_2027/paper/minigrid/hparam/results/centroid
                      CELLS="MiniGrid_8x8_16x16" ;;
            *) echo "unknown family '$fam'" >&2; exit 2 ;;
        esac
        # The arms with at least one finished trial.
        ARMS=$(for d in "$ROOT"/continual/{ga,nes}_sigma*; do
                   [ -d "$d" ] && ls "$d"/*/trial_*/training_metrics.json > /dev/null 2>&1 && basename "$d"; done)
        echo "=== $fam: arms [$ARMS]"
        [ -n "$ARMS" ] || { echo "    nothing finished under $ROOT/continual yet; skipped"; continue; }
        # Shared-runner columns (MiniGrid) to the gymnax names, in place, once.
        [ "$fam" = minigrid ] && $PY scripts/analysis/migrate_shared_runner_columns.py "$ROOT" | tail -1
        mkdir -p "$RESULTS"
        $PY scripts/analysis/behavioural_divergence.py \
            --runs_root "$ROOT" --methods $ARMS --envs $CELLS \
            --num_tasks 20 --episodes 20 --num_states 2000 --gpus "${GPUS:-3}" \
            --agent_source centroid --results_dir "$RESULTS" || exit 1
    done
fi

if has plot; then
    $PY scripts/analysis/plot_hparam_sigma.py --extract || exit 1
    if [ -d "$OVERLEAF/images/appendix" ]; then
        cp projects/iclr_2027/paper/visuals/final/appendix/hparam_sigma.pdf \
           projects/iclr_2027/paper/visuals/final/appendix/hparam_sigma_curves.pdf \
           "$OVERLEAF/images/appendix/"
        echo "=== figures copied to $OVERLEAF/images/appendix/ (commit and push them yourself)"
    fi
fi
echo "=== $(date -Is) done"
