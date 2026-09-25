#!/bin/bash
# ============================================================================
# Appendix: the effect of HYPERPARAMETERS, the UPDATE-COUNT half
# (gymnax, observation noise).
#
# Two scripts feed the hyperparameter appendix and they must not share a
# name, a tree or a process pattern:
#   queue_iclr_hparam.sh          the search-width half: GA / NES sigma and
#                                 PPO lr / entropy at x0.1 x0.3 x3 x10, arms
#                                 named by value, under runs_hparam/gymnax/
#   queue_iclr_hparam_updates.sh  THIS: how many updates a sub-task gets and
#                                 how far each moves, one tree a setting,
#                                 under runs_hparam/<setting>/gymnax/
# (The first launch of this script was called queue_iclr_hparam.sh and was
# killed by a pkill on that name from the other launch, 2026-09-19 20:1x.)
#
# The reported gymnax noise cells (runs_centroid, trials 1-5) run PPO at
# K = 10 epochs x M = 32 minibatches over a rollout of L = 50 steps on
# N = 2048 environments, and NES at sigma = 0.1, alpha = 0.05, P = 512. Each
# setting below changes ONE of those and nothing else. The budget is held at
# 3.072e9 env steps a run, 20 phases cycling 10 sub-tasks (TASK_PERIOD=10), so
# every arm still sees every switch at the same env step.
#
#   PPO -- how many gradient steps a sub-task gets, three ways:
#     epochs   K in {1, 3, 30}        gradient steps per update x K/10,
#                                      same data per update, each sample
#                                      reused K times
#     minibatches M in {4, 8, 128}    gradient steps per update x M/32 over
#                                      the SAME data and the same K passes:
#                                      only the number of steps taken over a
#                                      rollout changes (and their batch,
#                                      102400 / M). The cleanest reading of
#                                      "how many gradient updates a sub-task
#                                      gets": U x K x M = 60k / 120k / 1.92M
#                                      against the reported 480k
#     rollout  L in {10, 250}         updates per phase 7500 / 300 (1500
#                                      reported); the batch is N x L, so the
#                                      minibatch scales with it and so does the
#                                      GAE horizon
#     lr       alpha x {0.1, 10}      per-env base (Acrobot 1e-4, others 3e-4)
#
#   The minibatch axis is ALSO run on two more families, since no single value
#   is expected to be optimal across tasks (user, 2026-09-20):
#     actions_ppo_minibatches{4,8,128}    gymnax ACTION REVERSAL (runs_actions:
#                                          CartPole/Acrobot/MountainCar, tag
#                                          sigma1.0), reported M = 32
#     minigrid_ppo_minibatches{2,4,64}    MiniGrid 8x8/16x16 (runs_centroid/
#                                          minigrid), reported M = 16, so the
#                                          same x1/8, x1/4, x4 multipliers
#   and on HalfCheetah on CLUSTER: scripts/train/cluster/submit_cheetah_minibatches.sh.
#
#   NES -- the evolutionary counterparts:
#     pop      P in {128, 2048}       generations per phase 800 / 50 (200
#                                      reported), compute-matched
#     lr       x {0.25, 4}            0.0125 / 0.2
#   (NES sigma is the other script's axis, at x0.1 .. x10; not repeated here.)
#
# The reported setting is NOT re-run: runs_centroid trials 1..5 are the same
# seeds (BASE_SEED 42) and the same nested sub-task draw.
#
# Each setting is its own tree, projects/iclr_2027/runs_hparam/<setting>,
# with the usual gymnax/continual/<arm>/<cell>/trial_<k> shape underneath, so
# every analysis script that takes a run root reads it unchanged. The tag in
# the setting name is the value (epochs1, rollout250, pop128) or the
# multiplier of the reported value (lr10x).
#
# A rollout change must move RL_STEPS_PER_UPDATE with it -- that is what
# run_experiments.sh converts the NE switch interval into PPO updates with.
# The eval interval moves the other way so every run keeps ~3000 records: the
# figures resample on env steps, and 15000 NTK/dormancy probes a run is time.
#
# Slots: GPUS_ALL is split per launch, since launch.sh leases per launcher.
# The two sigma cells (CartPole/Acrobot at 1.0, MountainCar at 0.1) are two
# launchers run concurrently on disjoint GPUs; the PPO lr settings are three,
# one per env, because the base lr differs on Acrobot. run_condition skips
# finished trials, so re-running is safe -- but NOT while a trial of the
# same setting is still in flight (it is not skipped and would start twice),
# which is why ppo_epochs1 comes last by default: its first launch is still
# finishing its trials.
#
# Env: SETTINGS (default: all, PPO first), NUM_TRIALS (5), GPUS_ALL (default:
#      GPUs 3-7, 3 jobs each), DRY_RUN.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

NUM_TRIALS="${NUM_TRIALS:-5}"
export DRY_RUN="${DRY_RUN:-0}"
GPUS_ALL="${GPUS_ALL:-3 3 3 4 4 4 5 5 5 6 6 6 7 7 7}"
SETTINGS="${SETTINGS:-ppo_minibatches4 ppo_minibatches8 ppo_minibatches128 ppo_epochs3 ppo_epochs30 ppo_rollout10 ppo_rollout250 ppo_lr0.1x ppo_lr10x nes_pop128 nes_pop2048 nes_lr0.25x nes_lr4x ppo_epochs1}"
ROOT_BASE="projects/iclr_2027/runs_hparam"

# The reported values every multiplier is relative to.
declare -A PPO_LR=([CartPole-v1]=3e-4 [Acrobot-v1]=1e-4 [MountainCar-v0]=3e-4)
NES_SIGMA=0.1
NES_LR=0.05

slice() { echo "$GPUS_ALL" | tr ' ' '\n' | sed -n "$1,$2p" | tr '\n' ' '; }
NSLOT=$(echo "$GPUS_ALL" | wc -w)

# launch <setting> <envs> <sigma> <gpus> <arm> [BLOCK VARS...]
# The block is gymnax_continual unless BLOCK_NAME says otherwise (the
# MiniGrid settings pass minigrid_continual; its schedule is settings.py's,
# so the gymnax NUM_TASKS/TASK_PERIOD/SIGMAS below are ignored there).
launch() {
    local setting="$1" envs="$2" sigma="$3" gpus="$4" arm="$5"; shift 5
    local tag; tag="$(echo "$envs" | tr ' ' '+')"
    env "$@" NUM_TRIALS="$NUM_TRIALS" PROJECT_ROOT="$ROOT_BASE/$setting" \
        ENVS="$envs" SIGMAS="$sigma" NUM_TASKS=20 TASK_PERIOD=10 \
        GPUS="$gpus" LOG_DIR="logs/launch_hparam_${setting}_${tag}" \
        bash scripts/train/launch.sh "${BLOCK_NAME:-gymnax_continual}" "$arm"
}

# Two launchers per setting: 2/3 of the slots for the two sigma-1.0 cells,
# 1/3 for MountainCar (10 + 5 jobs on 15 slots).
two_cells() {
    local setting="$1" arm="$2"; shift 2
    local split=$((NSLOT * 2 / 3))
    launch "$setting" "CartPole-v1 Acrobot-v1" 1.0 "$(slice 1 $split)" "$arm" "$@" &
    launch "$setting" "MountainCar-v0" 0.1 "$(slice $((split + 1)) $NSLOT)" "$arm" "$@" &
    wait
}

for setting in $SETTINGS; do
    echo "=== $(date -Is) $setting -> $ROOT_BASE/$setting ==="
    case "$setting" in
        ppo_epochs*)
            two_cells "$setting" ppo RL_EXTRA_ARGS="--num_epochs ${setting#ppo_epochs}" ;;
        ppo_minibatches*)
            two_cells "$setting" ppo RL_EXTRA_ARGS="--num_minibatches ${setting#ppo_minibatches}" ;;
        # The ACTION-REVERSAL family (runs_actions): all three envs in one
        # cell group at the reported sigma tag 1.0 (a name there, not a
        # setting), TASK_TYPE=actions, the same 20 phases at period 10.
        actions_ppo_minibatches*)
            M=${setting#actions_ppo_minibatches}
            launch "$setting" "CartPole-v1 Acrobot-v1 MountainCar-v0" 1.0 "$GPUS_ALL" ppo \
                TASK_TYPE=actions RL_EXTRA_ARGS="--num_minibatches $M" ;;
        # MiniGrid 8x8/16x16 (runs_centroid/minigrid): the shared runner,
        # reported M = 16 (K = 4, 3072 updates a phase), override through the
        # block's MINIGRID_EXTRA hook. One cell, so the slots are capped at
        # NUM_TRIALS jobs anyway.
        minigrid_ppo_minibatches*)
            M=${setting#minigrid_ppo_minibatches}
            BLOCK_NAME=minigrid_continual \
            launch "$setting" "MiniGrid_8x8_16x16" 1.0 "$GPUS_ALL" ppo \
                MINIGRID_EXTRA="--ppo_override num_minibatches=$M" ;;
        ppo_rollout*)
            L=${setting#ppo_rollout}
            # ~3000 records a run at any L: reported is every 10 of 30000 updates.
            EV=$(( 10 * 50 / L )); [ "$EV" -lt 1 ] && EV=1
            two_cells "$setting" ppo RL_STEPS_PER_UPDATE=$((2048 * L)) \
                RL_EXTRA_ARGS="--num_steps $L --eval_interval $EV" ;;
        ppo_lr*)
            mult=${setting#ppo_lr}; mult=${mult%x}
            third=$((NSLOT / 3))
            i=0
            for env in CartPole-v1 Acrobot-v1 MountainCar-v0; do
                lr=$(python3 -c "print(f'{${PPO_LR[$env]} * $mult:.3g}')")
                sig=1.0; [ "$env" = MountainCar-v0 ] && sig=0.1
                launch "$setting" "$env" "$sig" "$(slice $((i * third + 1)) $(((i + 1) * third)))" ppo \
                    RL_EXTRA_ARGS="--learning_rate $lr" &
                i=$((i + 1))
            done
            wait ;;
        # The RL FAMILY at a multiple of its per-task learning rate: the
        # follow-up to the ppo_lr axis, which found PPO at x0.1 far better on
        # Acrobot and CartPole. The RL arms share PPO's settings, so the
        # question "is the NE-vs-RL ordering a property of the learning rate"
        # needs all of them at the new rate. PPO itself is NOT re-run: its
        # trials are runs_hparam/ppo_lr<m>x, linked into this tree so the
        # four arms read as one. Tree: runs_hparam/rl_lr<m>x.
        rl_lr*)
            mult=${setting#rl_lr}; mult=${mult%x}
            src="$ROOT_BASE/ppo_lr${mult}x/gymnax/continual/ppo"
            dst="$ROOT_BASE/$setting/gymnax/continual"
            if [ -d "$src" ] && [ "$DRY_RUN" = 0 ]; then
                mkdir -p "$dst"
                [ -e "$dst/ppo" ] || ln -s "$(realpath --relative-to="$dst" "$src")" "$dst/ppo"
            fi
            third=$((NSLOT / 3))
            i=0
            for env in CartPole-v1 Acrobot-v1 MountainCar-v0; do
                lr=$(python3 -c "print(f'{${PPO_LR[$env]} * $mult:.3g}')")
                sig=1.0; [ "$env" = MountainCar-v0 ] && sig=0.1
                launch "$setting" "$env" "$sig" "$(slice $((i * third + 1)) $(((i + 1) * third)))" \
                    "trac redo cchain" RL_EXTRA_ARGS="--learning_rate $lr" &
                i=$((i + 1))
            done
            wait ;;
        nes_pop*)
            P=${setting#nes_pop}
            # Compute-matched: generations per phase scale as 512/P.
            two_cells "$setting" nes NE_POP_SIZE=$P TASK_INTERVAL=$((200 * 512 / P)) ;;
        nes_sigma*)
            mult=${setting#nes_sigma}; mult=${mult%x}
            two_cells "$setting" nes ES_EXTRA_ARGS="--sigma $(python3 -c "print($NES_SIGMA * $mult)")" ;;
        nes_lr*)
            mult=${setting#nes_lr}; mult=${mult%x}
            two_cells "$setting" nes ES_EXTRA_ARGS="--learning_rate $(python3 -c "print($NES_LR * $mult)")" ;;
        *) echo "FATAL: unknown setting $setting" >&2; exit 1 ;;
    esac
done

echo "=== $(date -Is) done ==="
echo "failures:"
grep -hv 'exit=0' logs/launch_hparam_*_*/status.tsv 2>/dev/null || echo "  none"
