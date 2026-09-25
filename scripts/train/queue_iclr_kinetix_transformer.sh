#!/bin/bash
# ============================================================================
# KINETIX WITH A TRANSFORMER. The paper's Kinetix grid again, on Kinetix's
# symbolic-entity observation and its transformer network instead of pixels
# and the conv net -- does the result hold when the policy is a transformer?
#
# Nothing here is a new trainer or a new setting. `--observation entity`
# (source/studies/kinetix/cli.py) swaps the observation and, through it, the
# network (source/envs/kinetix.py:build_policy -> networks.
# KinetixTransformerPolicy, Kinetix's `ActorCriticTransformer` actor half,
# verified logit-for-logit against it). Arms, budget (200 gen x 512 x 1 x 128
# = 1600 updates x 128 x 64 a level), cells, schedule and seeds (41 + trial)
# are the pixel runs' exactly, through the same two queue scripts.
#
# The arms are the paper's Kinetix arms by their REAL names:
#     GA  = ga_focus_explore_nox     ES = es_hold
#     ppo trac redo cchain pbt       (pbt = PBT-PPO, N = 8)
#
# STAGES, run in order, each to completion before the next:
#     ne_noncont   GA, ES    x 20 stationary levels x NUM_TRIALS
#     rl_noncont   RL arms   x 20 stationary levels x NUM_TRIALS
#     ne_cont      GA, ES    x Kinetix20 chain      x NUM_TRIALS
#     rl_cont      RL arms   x Kinetix20 chain      x NUM_TRIALS
# The continual NE chains write a resume point every level
# (--checkpoint_every 200) and the RL chains every 2000 updates, as the
# paper's chains did.
#
# USAGE
#   STAGES=ne_noncont nohup bash scripts/train/queue_iclr_kinetix_transformer.sh \
#       > logs/kinetix_tf.log 2>&1 &
#   STAGES="rl_noncont ne_cont rl_cont" nohup bash ... &
#
# Env: STAGES, ROOT, NUM_TRIALS (3), NE_ARMS, RL_ARMS, NE_GPUS, RL_GPUS,
#      PBT_GPUS, DRY_RUN.
# NE_GPUS / RL_GPUS are launch.sh token lists (repeat a card to put two jobs
# on it). PBT runs eight learners in one process and gets its own card
# (PBT_GPUS), after the other RL arms of the stage.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ROOT="${ROOT:-projects/iclr_2027/runs_kinetix_tf}"
NUM_TRIALS="${NUM_TRIALS:-3}"
STAGES="${STAGES:-ne_noncont rl_noncont ne_cont rl_cont}"
NE_ARMS="${NE_ARMS:-ga_focus_explore_nox es_hold}"
RL_ARMS="${RL_ARMS:-ppo trac redo cchain}"
PBT_ARMS="${PBT_ARMS:-pbt}"
ALL8="0 1 2 3 4 5 6 7"
NE_GPUS="${NE_GPUS:-$ALL8 $ALL8}"
RL_GPUS="${RL_GPUS:-$ALL8 $ALL8}"
PBT_GPUS="${PBT_GPUS:-$ALL8}"
DRY_RUN="${DRY_RUN:-0}"
export ROOT NUM_TRIALS DRY_RUN
# block_kinetix only runs arms named in METHOD_ORDER; the variants are not in
# its default list.
export METHOD_ORDER="$NE_ARMS $RL_ARMS $PBT_ARMS"

if ! grep -q "'--observation'" source/studies/kinetix/cli.py; then
    echo "FATAL: this checkout's Kinetix CLI has no --observation." >&2
    exit 1
fi

stage() {   # stage <noncont|cont> <arms> <gpus> <extra flags>
    local phase="$1" arms="$2" gpus="$3" extra="$4" script
    if [ "$phase" = noncont ]; then
        script=scripts/train/queue_iclr_kinetix_noncontinual.sh
    else
        script=scripts/train/queue_iclr_kinetix_continual.sh
    fi
    echo "=== $(date -Is) $phase | $arms | gpus [$gpus]"
    ARMS="$arms" GPUS="$gpus" SKIP_STATIONARY_CHECK=1 \
        KINETIX_EXTRA="--observation entity $extra" bash "$script"
}

for s in $STAGES; do
    case "$s" in
        ne_noncont) stage noncont "$NE_ARMS" "$NE_GPUS" "" ;;
        rl_noncont) stage noncont "$RL_ARMS" "$RL_GPUS" ""
                    stage noncont "$PBT_ARMS" "$PBT_GPUS" "" ;;
        ne_cont)    stage cont "$NE_ARMS" "$NE_GPUS" "--checkpoint_every 200" ;;
        rl_cont)    stage cont "$RL_ARMS $PBT_ARMS" "$PBT_GPUS" \
                          "--checkpoint_every 2000" ;;
        *) echo "unknown stage $s" >&2; exit 1 ;;
    esac
done
echo "=== $(date -Is) all stages done"
