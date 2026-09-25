#!/bin/bash
# ============================================================================
# Appendix: the effect of HYPERPARAMETERS (gymnax, observation noise).
#
# The paper compares every method at ONE setting each (Appendix tables
# hyper_ne / hyper_rl), tuned on the stationary task. This sweep asks whether
# the conclusions of the continual comparison -- ES has the best stability /
# plasticity trade-off, the GA is the most plastic, PPO loses plasticity --
# depend on that choice. ONE hyperparameter at a time is moved over two
# orders of magnitude around the reported value, x0.1 x0.3 x3 x10, and
# everything else is the reported run's (20 phases x 200 generations, ten
# sub-tasks visited twice, the same seeds, the same sub-task draw, diversity
# tracked):
#
#     GA    mutation width  sigma   0.5   -> 0.05 0.15 1.5 5.0
#     ES    search width    sigma   0.1   -> 0.01 0.03 0.3 1.0   (NES, lr 0.05 kept)
#     PPO   learning rate   alpha   3e-4  -> 3e-5 1e-4 1e-3 3e-3  (CartPole, MountainCar)
#                                   1e-4  -> 1e-5 3e-5 3e-4 1e-3  (Acrobot)
#     PPO   entropy coef.   beta    0.01  -> 0.001 0.003 0.03 0.1
#
# The reported value is NOT re-run: its trials are the same seeds in
# projects/iclr_2027/runs_centroid, and the figure reads them from the
# paper's forgetting pass (paper/gymnax/noise/10task/results/centroid).
#
# Cells: the ones of the main continual figure -- CartPole / Acrobot at
# offset sigma 1.0, MountainCar at 0.1. The value is part of the arm name
# (run_experiments.sh, block_gymnax_continual, "HYPERPARAMETER ARMS"), so one
# launcher per cell carries every arm in one job list:
#
#     projects/iclr_2027/runs_hparam/gymnax/continual/<arm>/<cell>/trial_<k>
#
# 16 arms x 5 trials a cell = 240 jobs, ~1.1 h each at four to a GPU. The NE
# arms are listed first, so they are queued first. run_condition skips a
# finished trial, so this is safe to restart.
#
# FAMILY selects the cells (2026-09-20):
#     noise     (default) the three cells above -> runs_hparam/gymnax
#     actions   the same three environments under action reversal (two
#               alternating sub-tasks, TASK_TYPE=actions, offset name 1.0)
#               -> runs_hparam_actions/gymnax; NE arms only, same widths
#     minigrid  the 8x8 / 16x16 room chain on the shared runner
#               -> runs_hparam_minigrid/minigrid; GA width 0.01 -> 0.001 0.003
#               0.03 0.1, ES width 0.1 -> 0.01 0.03 0.3 1.0 (--ne_override);
#               one launcher, GPUS_A
#
# Env: NUM_TRIALS (5), GPUS_A / GPUS_B / GPUS_C (one slot list per cell;
#      launch.sh leases are per launcher, so the three lists must be DISJOINT
#      -- default GPUs 3-7, four slots each, 7 + 7 + 6), NE_ONLY=1 / RL_ONLY=1
#      to queue one family, DRY_RUN=1.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

FAMILY="${FAMILY:-noise}"
ROOT=projects/iclr_2027/runs_hparam
NUM_TRIALS="${NUM_TRIALS:-5}"
export DRY_RUN="${DRY_RUN:-0}"
export TRACK_DIVERSITY=1 NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 PROJECT_ROOT="$ROOT"

NE_ARMS="ga_sigma0.05 ga_sigma0.15 ga_sigma1.5 ga_sigma5.0 nes_sigma0.01 nes_sigma0.03 nes_sigma0.3 nes_sigma1.0"
ENT_ARMS="ppo_ent0.001 ppo_ent0.003 ppo_ent0.03 ppo_ent0.1"
LR_ARMS_3E4="ppo_lr3e-05 ppo_lr0.0001 ppo_lr0.001 ppo_lr0.003"     # reported 3e-4
LR_ARMS_1E4="ppo_lr1e-05 ppo_lr3e-05 ppo_lr0.0003 ppo_lr0.001"     # reported 1e-4

arms() {  # <lr arm list>
    local ne="$NE_ARMS" rl="$1 $ENT_ARMS"
    [ "${RL_ONLY:-0}" = 1 ] && ne=""
    [ "${NE_ONLY:-0}" = 1 ] && rl=""
    echo $ne $rl
}

case "$FAMILY" in
    noise) ;;
    actions)
        ROOT=projects/iclr_2027/runs_hparam_actions
        export TASK_TYPE=actions PROJECT_ROOT="$ROOT" NE_ONLY=1 ;;
    minigrid)
        ROOT=projects/iclr_2027/runs_hparam_minigrid
        echo "=== $(date -Is) hyperparameter sweep (MiniGrid) -> $ROOT ==="
        NUM_TRIALS="$NUM_TRIALS" PROJECT_ROOT="$ROOT" GPUS="${GPUS_A:-0 0 1 1 2 2}" \
          ENVS=MiniGrid_8x8_16x16 \
          LOG_DIR="logs/launch_hparam_minigrid" \
          bash scripts/train/launch.sh minigrid_continual \
            ga_sigma0.001 ga_sigma0.003 ga_sigma0.03 ga_sigma0.1 \
            nes_sigma0.01 nes_sigma0.03 nes_sigma0.3 nes_sigma1.0
        echo "=== $(date -Is) done ==="
        grep -hv 'exit=0' logs/launch_hparam_minigrid/status.tsv 2>/dev/null || echo "  no failures"
        exit 0 ;;
    *) echo "unknown FAMILY '$FAMILY'" >&2; exit 2 ;;
esac

go() {  # <env> <offset sigma> <gpu slots> <lr arm list>
    NUM_TRIALS="$NUM_TRIALS" ENVS="$1" SIGMAS="$2" GPUS="$3" \
      LOG_DIR="logs/launch_hparam${TAG}_${1%%-*}" \
      bash scripts/train/launch.sh gymnax_continual $(arms "$4") &
}

MCAR_SIGMA=0.1; TAG=""
[ "$FAMILY" = actions ] && { MCAR_SIGMA=1.0; TAG="_actions"; }
echo "=== $(date -Is) hyperparameter sweep ($FAMILY) -> $ROOT ==="
go CartPole-v1    1.0         "${GPUS_A:-3 3 3 3 4 4 4}" "$LR_ARMS_3E4"
go Acrobot-v1     1.0         "${GPUS_B:-4 5 5 5 5 6 6}" "$LR_ARMS_1E4"
go MountainCar-v0 $MCAR_SIGMA "${GPUS_C:-6 6 7 7 7 7}"   "$LR_ARMS_3E4"
wait

echo "=== $(date -Is) done ==="
echo "failures:"
grep -hv 'exit=0' logs/launch_hparam${TAG}_{CartPole,Acrobot,MountainCar}/status.tsv 2>/dev/null || echo "  none"
