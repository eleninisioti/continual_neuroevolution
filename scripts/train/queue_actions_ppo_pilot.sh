#!/bin/bash
# ============================================================================
# ACTION-REVERSAL PILOT: the first gymnax sub-task family with no generalist.
#
# Into projects/iclr_2027/runs_actions. PPO only, all three envs, 5 trials,
# CartPole first because launch.sh orders its job list by env.
#
# WHY THIS FAMILY EXISTS. The two families the paper has both turned out to be
# COMPATIBLE -- one policy covers every sub-task, so a switch is absorbed
# rather than relearned:
#
#   physics  standard range: PPO pinned at 500 through all 20 sub-tasks, and
#            zero-shot on an unseen body 438 against an end-of-sub-task 493.
#            Widened to 0.25x-4.0x the zero-shot dip becomes real (170 at
#            2.66x, 191 at 0.30x) but PPO is back at 500 within TEN updates and
#            by sub-task 3 zero-shots the next body at exactly 500 -- it has
#            become a generalist over pole length. Widening stretched the
#            transient; it created no interference.
#   noise    conflicting enough that PPO collapses, but the conflict is in the
#            OBSERVATION, so it is confounded with "the input distribution
#            moved" and a wide enough offset is simply a different input.
#
# Under `actions` sub-task i reverses the action order (a -> n-1-a) and leaves
# the observation and the body untouched. For a memoryless policy of the
# observation -- the NE MLP and PPO's actor alike -- the two regimes demand
# OPPOSITE outputs at the same input, so there is provably no single policy
# that scores on both. A method cannot interpolate its way across this
# boundary; it can only relearn. That is the property both other families
# lack.
#
# THE SEQUENCE ALTERNATES AND IS NOT DRAWN FROM THE TRIAL SEED. A reversal flag
# has two states, so sampling it would give runs of consecutive sub-tasks in
# the same regime -- boundaries at which nothing changes -- and a different
# number of real switches per trial. Alternation makes every one of the 19
# boundaries a real reversal and gives every trial the same number, which is
# what CLAUDE.md rule (c) asks for. Sub-task 0 is the stock order, so it is the
# noncontinual experiment exactly as under the other two families. See
# `action_flip_sequence` in source/utils/task_sequence.py.
#
# NO CUE, DELIBERATELY. `TaskSpec`'s `actions_cue` adds the trial's observation
# offset to the reversed sub-task, which makes the regime identifiable from the
# observation and a generalist possible again. That is a different and strictly
# easier experiment. Run it as the contrast IF the no-cue version turns out to
# be so hard that nothing moves at all.
#
# WHAT THE SMOKE TEST ALREADY SHOWS, and why this is worth the cards: a policy
# at 500 on the stock order scores 9.0 zero-shot on the reversed one, and after
# training on reversed scores 8.9 back on stock. Both directions are total. The
# open question this pilot answers is what happens over 1500 updates per
# sub-task and 20 sub-tasks: does PPO relearn each regime in full every time
# (no forgetting problem, just wasted compute), degrade run over run (loss of
# plasticity, the interesting result), or collapse to a hedging policy that
# does neither?
#
# PPO ONLY. The NE trainers (train_{GA,ES,DNS}_gymnax_continual.py) do NOT yet
# have an `actions` branch -- only train_RL_gymnax_continual.py does -- and
# run_experiments.sh passes --task_type to ES/NES, so queueing an NE arm here
# exits 2. Add the same branch to them before any compute-matched sweep; this
# pilot is a go/no-go on the family, not a cell of the paper.
#
# CONTENTION. A cluster launcher and the widened-physics pilot are already
# running here and launch.sh leases GPUs PER LAUNCHER. This takes 3 slots on
# card 0, which is the idle one (26% while the rest are at 93-100%). Override
# with GPUS=.
#
#   bash scripts/train/queue_actions_ppo_pilot.sh
#
# Env: ARMS (default ppo), ENVS (default all three, CartPole first),
#      GPUS, NUM_TRIALS (default 5).
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ ! -f "$REPO_ROOT/pyproject.toml" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
cd "$REPO_ROOT"

ARMS="${ARMS:-ppo}"
ENVS="${ENVS:-CartPole-v1 Acrobot-v1 MountainCar-v0}"
ROOT=projects/iclr_2027/runs_actions
TRIALS="${NUM_TRIALS:-5}"
export GPUS="${GPUS:-0 0 0}"

mkdir -p "$ROOT/gymnax"

echo "=== $(date -Is) action-reversal pilot ==="
echo "    root  : $ROOT"
echo "    arms  : $ARMS"
echo "    envs  : $ENVS"
echo "    gpus  : $GPUS"
echo "    trials: 1..$TRIALS"
echo "    tasks : 20 at period 10, flags 0,1,0,1,... (sub-task 0 = stock order)"

NUM_TRIALS="$TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  ENVS="$ENVS" \
  TASK_TYPE=actions NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 SIGMAS=1.0 \
  LOG_DIR=logs/launch_actions \
  bash scripts/train/launch.sh gymnax_continual $ARMS

echo "=== $(date -Is) done ==="
echo "failures:"
grep -hv 'exit=0' logs/launch_actions/status.tsv 2>/dev/null || echo "  none"
