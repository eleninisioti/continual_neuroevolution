#!/bin/bash
# ============================================================================
# THE RETENTION HEAD-TO-HEAD: do the NE arms retain a regime better than PPO?
#
# Into projects/iclr_2027/runs_actions_short. CartPole action reversal, five
# arms, 2 trials, at ONE TENTH the paper budget. A DIAGNOSTIC, not a paper cell
# -- it exists to decide whether the NE arms are worth reporting on this family
# at all, before 240 jobs are spent on the other machine.
#
# WHY A TENTH. The question is about the ZERO-SHOT column, which is read at
# every sub-task boundary, so it needs the full 20-sub-task / period-10
# structure but NOT the full per-sub-task budget. PPO recovers from a reversal
# in a median of 10 updates (measured, 38/38 reversals, p90 16, max 31), so 150
# updates per sub-task is still a 15x surplus for it, and 20 generations is
# enough for the NE arms to clear CartPole from a warm start. Cutting the
# per-sub-task budget rather than the sub-task count is what keeps the thing
# being measured intact.
#
#   full   200 gens / 1500 updates per sub-task, 3.072e9 steps
#   here    40 gens /  300 updates per sub-task, 6.144e8 steps
#
# COMPUTE MATCHING IS THE LAUNCHER'S JOB AND IT IS NOT OPTIONAL HERE. TASK_INTERVAL
# is in NE GENERATIONS and run_experiments.sh derives the RL interval from it as
# pop*evals*episode_length*interval/102400 = 7.5x, so TASK_INTERVAL=20 gives PPO
# 150 updates and every arm meets its boundaries at the same env step
# (CLAUDE.md rule (c)). Do not set the RL interval by hand.
#
# WHAT IS BEING COMPARED, AND WHAT WOULD FALSIFY IT. At each boundary the saved
# agent is scored on the NEW regime before anything trains on it. Sub-task k
# for even k is the STOCK order, which sub-task k-2 already trained on, so its
# zero-shot is a RETENTION number: how much of a regime survives one sub-task of
# its opposite. PPO's is 8.9 +/- 0.2, floor-level, at every boundary. An NE arm
# is worth reporting on this family only if it beats that.
#
# THE TRAP THIS RUN MUST NOT FALL INTO. A method that never learns a regime has
# perfect retention trivially -- it has nothing to lose. The 10-generation smoke
# test hit exactly this: NES and DNS ended sub-task 0 at 126 and 153 on a task
# whose threshold is 475, and their zero-shot of 9.4 measured nothing. So read
# the two columns TOGETHER: an arm's retention number only counts if its
# END-OF-SUB-TASK number shows it had learned the regime in the first place.
# That is the same artifact as the frozen GA archive, and it is why this script
# prints both.
#
#   bash scripts/train/queue_actions_retention.sh
#
# Env: ARMS, ENVS, GPUS, NUM_TRIALS (default 2).
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ ! -f "$REPO_ROOT/pyproject.toml" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
cd "$REPO_ROOT"

ARMS="${ARMS:-ga dns_gaussian es nes ppo}"
ENVS="${ENVS:-CartPole-v1 Acrobot-v1 MountainCar-v0}"
ROOT=projects/iclr_2027/runs_actions_short
TRIALS="${NUM_TRIALS:-3}"
# NE GENERATIONS per sub-task. run_experiments.sh derives the RL interval from
# it as 7.5x, so 40 here is 300 PPO updates -- thirty times PPO's measured
# 10-update recovery, so PPO is in no way handicapped by the shortening, while
# the NE arms get twice the generations of the first attempt. Undertrained NE is
# the one failure mode that would make this whole comparison unreadable (an arm
# that never learned a regime cannot lose it), so the budget is set from the NE
# side and PPO's surplus is whatever falls out.
INTERVAL="${TASK_INTERVAL:-40}"
# Four slots. The box is already carrying ~44 gymnax jobs from the cluster
# sweep and the PPO actions pilot, so this takes a small share rather than a
# fair one; launch.sh leases per launcher, not globally.
export GPUS="${GPUS:-4 4 4 4 5 5 5 5 6 6 6 6 2 2}"

mkdir -p "$ROOT/gymnax"

echo "=== $(date -Is) action-reversal RETENTION head-to-head ==="
echo "    root  : $ROOT"
echo "    arms  : $ARMS"
echo "    envs  : $ENVS"
echo "    budget: $INTERVAL gens / $((INTERVAL*15/2)) PPO updates per sub-task"

NUM_TRIALS="$TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  ENVS="$ENVS" \
  TASK_TYPE=actions NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL="$INTERVAL" SIGMAS=1.0 \
  LOG_DIR=logs/launch_actions_short \
  bash scripts/train/launch.sh gymnax_continual $ARMS

echo "=== $(date -Is) done ==="
grep -hv 'exit=0' logs/launch_actions_short/status.tsv 2>/dev/null || echo "failures: none"
