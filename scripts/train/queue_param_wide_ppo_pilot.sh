#!/bin/bash
# ============================================================================
# WIDENED PHYSICS PILOT: does PPO still coast when the bodies change more?
#
# Into projects/iclr_2027/runs_param_wide. PPO only, CartPole only, 5 trials.
# This is a PILOT, not a cell of the paper: it asks one question and the answer
# decides whether the physics family is worth a full eight-arm sweep.
#
# THE QUESTION. On the standard physics range PPO is pinned at 500 through all
# 20 sub-tasks -- zero-shot on an unseen body 438 against an end-of-sub-task
# 493, while its policy dormancy climbs 13% -> 48% and its value dormancy
# 41% -> 75% exactly as it does in the observation-noise cell where it collapses
# to 247. Nothing about the physics family is hard for anything, so it separates
# no methods. Widening the range is the cheapest thing that might fix that.
#
# THE RANGE, AND WHY THIS ONE. `analysis/physics_zero_shot.py` scored stock-body
# NES specialists over 0.25x-4x on 2026-09-05 and that sweep is what fixed the
# current defaults. On CartPole `length` it found:
#
#     0.25x   284      2x   38.6      (stock specialists, 8 trials, 500 = solved)
#
# so the measured span 0.25x-4x is where a stock specialist demonstrably fails
# at BOTH ends. The standard range 0.5x-2x is doubled in log width:
#
#     standard   0.5 - 2.0    log width 1.39
#     wide       0.25 - 4.0   log width 2.77
#
# CARTPOLE AND NOT THE OTHER TWO, and this is a measured constraint, not a
# preference:
#
#   Acrobot  `mass` has almost no headroom. Same sweep: a specialist trained at
#            1.15x reaches -74.9, at 1.25x -85.5, at 1.5x -99.4, at 2x -138.
#            Every widening trades "PPO fails" against "the body is not
#            solvable", and the two are not distinguishable from one run.
#   MountainCar is NESTED. Every physics rescaling of it came out nested on
#            2026-09-07 -- "push along the velocity" is physics-agnostic, so a
#            policy that pumps harder solves every weaker car. A wider gravity
#            range is a wider nest, not a harder problem.
#
# Widen Acrobot with ENVS="Acrobot-v1" PARAM_RANGE="0.5 2.0" if you want it
# anyway; read its result knowing the specialist ceiling there is about -138.
#
# WHAT THIS PILOT CANNOT TELL YOU. If PPO fails at 4x, that is either "hard but
# learnable" (interesting) or "unsolvable body" (not). This run does not carry
# a from-scratch specialist at 4x, so it cannot separate them. What it does
# carry is a warm start plus 1500 PPO updates = 1.536e8 env steps PER SUB-TASK,
# which is two orders of magnitude more than a from-scratch PPO needs for stock
# CartPole -- so an end-of-sub-task score that stays low is at least not a
# budget artifact. Run the specialist control only if the pilot fails; there is
# no point paying for it while PPO is still at 500.
#
# THE RANGE IS PASSED, NOT EDITED INTO THE TABLE. PARAM_RANGE overrides
# GYMNAX_PHYSICS_TASKS per launch and each run records its own `param_range` in
# its config, so the standard-range tree stays reproducible from the same
# source. Nothing in source/ changes for this pilot.
#
# A SEPARATE ROOT, deliberately. The cell name is `CartPole_v1_sigma1.0` under
# both ranges -- the `_sigma` suffix is a naming convention the analysis scripts
# split on, not a setting -- so a wide run and a standard run would be
# indistinguishable inside one tree and `run_condition` would skip whichever
# landed second.
#
# CONTENTION. A cluster launcher is already running here and launch.sh leases
# GPUs PER LAUNCHER, so it has all 8. This pilot takes 2 slots on the two
# least-loaded cards and will contend for compute; memory is not at issue
# (2.4 GB used of 49 GB per card, a gymnax job is ~600 MiB). Override with GPUS=.
#
#   bash scripts/train/queue_param_wide_ppo_pilot.sh
#
# Env: ARMS (default ppo), ENVS (default CartPole-v1), PARAM_RANGE
#      (default "0.25 4.0"), GPUS, NUM_TRIALS (default 5).
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ ! -f "$REPO_ROOT/pyproject.toml" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
cd "$REPO_ROOT"

ARMS="${ARMS:-ppo}"
ENVS="${ENVS:-CartPole-v1}"
ROOT=projects/iclr_2027/runs_param_wide
TRIALS="${NUM_TRIALS:-5}"
export PARAM_RANGE="${PARAM_RANGE:-0.25 4.0}"

# Two slots on the two least-loaded cards. See the contention note above.
export GPUS="${GPUS:-7 1}"

mkdir -p "$ROOT/gymnax"

echo "=== $(date -Is) widened physics pilot ==="
echo "    root  : $ROOT"
echo "    arms  : $ARMS"
echo "    envs  : $ENVS"
echo "    range : $PARAM_RANGE   (standard is 0.5 2.0)"
echo "    gpus  : $GPUS"
echo "    trials: 1..$TRIALS"

NUM_TRIALS="$TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  ENVS="$ENVS" \
  TASK_TYPE=param NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 SIGMAS=1.0 \
  LOG_DIR=logs/launch_param_wide \
  bash scripts/train/launch.sh gymnax_continual $ARMS

echo "=== $(date -Is) done ==="
echo "failures:"
grep -hv 'exit=0' logs/launch_param_wide/status.tsv 2>/dev/null || echo "  none"
