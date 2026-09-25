#!/bin/bash
# ============================================================================
# THE CHEETAH STATIONARY BLOCK. Run this on the second server.
#
# 80 trials: 8 arms x 1 cell x 10 trials, every one a full-budget stationary
# run of brax's halfcheetah. It is the control the two cheetah continual cells
# are read against -- a dormancy, rank or retention number only says something
# about task CHANGES if the same measurement on a run of the same length with
# no change does something different -- and it is also what says whether an arm
# can run at all before anything is asked about carrying ten grounds.
#
# THE GRID
#
#     cell    cheetah         the unperturbed body throughout (schedule task0)
#     arms    ga es nes dns_gaussian ppo trac redo cchain
#     trials  1..10
#
# ONE STATIONARY CELL SERVES BOTH FAMILIES. Sub-task 0 is the unperturbed body
# under either kind of perturbation (a zero observation offset, a x1.0 friction
# multiplier); the sub-task vectors are drawn off the TRIAL rather than the
# run's key stream, so a `task0` run never touches the difference. It is filed
# under the friction family so its nine idle evaluation columns are zero-shot
# friction transfer -- which on this body is worth reading on its own: in the
# probe a solved policy (t0 4531) scored 4488 on one multiplier and 706 on
# another, so the ground bites hard zero-shot on a two-footed planar body in a
# way it did not on the mujoco_playground cheetah.
#
# THIS IS A DIFFERENT BODY FROM runs_repro2/mujoco AND ITS NUMBERS ARE ON A
# DIFFERENT SCALE. That tree is dm_control's CheetahRun through
# mujoco_playground: a ONE-sided `rewards.tolerance` at _RUN_SPEED 10, per-step
# in [0, 1], episode capped at 1000. This is brax `halfcheetah` on MJX with
# `TargetSpeedWrapper`: a TWO-sided Gaussian peaking AT the target, weight 5,
# no healthy bonus, no termination -- an untrained network scores ~475-690 and
# the ceiling is ~5000. Carrying a second simulator for one body is the
# duplication CLAUDE.md (a) is about, which is why the move was made; the price
# is that no cheetah number from before 2026-09-08 is comparable, and this
# block is what replaces them.
#
# THE GATE HAS PASSED ON THE ONE QUESTION THAT COULD HAVE STOPPED IT.
# `target_speed` 10.0 is dm_control's constant for dm_control's xml and
# `source/envs/mjx.py` flagged it PROVISIONAL on this one -- if nothing could
# reach it, every sub-task would sit in the same flat low-credit band and
# nothing would discriminate. Measured 2026-09-09 by
# scripts/train/probe_cheetah_stationary.sh: PPO reaches 4593 and 4649 across
# two seeds, ~92% of the ceiling, inside 4% of the budget. The target is
# reachable and the block is safe to queue.
#
# TWO SETTINGS ARE STILL ASSUMED RATHER THAN MEASURED, and both are cheap to
# undo:
#
#   NE sigma      `ga` and `dns_gaussian` share a mutation width of 0.1, read
#                 off runs_repro2/mujoco's own configs. Sigma is a property of
#                 the BODY and the policy parameterisation -- a (128,128) tanh
#                 MLP over 17 observations and 6 actuators -- and neither
#                 changed with the simulator, so it should transfer. The probe
#                 runs `ga_ant_sigma` (the ant's 0.01) as the contrast that
#                 says so instead of assuming it; if 0.01 wins, exactly two
#                 arms here re-run (~30 job-hours). Hold them with
#                 ARMS="es nes ppo trac redo cchain" if you would rather wait
#                 for that answer.
#   NES step      `nes` takes OpenES's sigma at half its step (0.04 / 0.005),
#                 the relation the ant's swept row turned out to have. repro2
#                 has NO nes arm on this body. If the NES curve looks unlike
#                 the OpenES one, that is the first thing to sweep.
#
# THE PPO ROLLOUT SHAPE IS NOT ONE OF THEM, and deliberately. repro2's cheetah
# PPO differs from `_MJX_PPO` in lr (1e-4 vs 3e-4), update size (81,920 vs
# 10,240 env steps) and reward scaling (0.1 vs 10) -- but copying those numbers
# would match their NAMES rather than their meaning, since `reward_scaling` 0.1
# against a reward capped at 1/step is a different object from 0.1 against one
# capped at 5/step. And a STATIONARY cell cannot choose between the shapes
# anyway: PPO saturates at ~4600 within 4% of the budget, so both finish at the
# ceiling. That question belongs to the continual block -- does a shape
# re-solve a sub-task so fast that no degradation is measurable, which is the
# trap block_brax_continual hit on the ant -- and it is asked there. This block
# runs the shared `_MJX_PPO` shape, which is also the ant's, so one shape
# spans both bodies.
#
# COMPUTE-MATCHED, and asserted rather than asserted-in-a-comment: every arm
# gets 4.9152e8 environment steps, as 320 generations x 512 x 3 x 1000 for the
# NE arms and 48,000 updates x 512 x 20 for the RL ones, cut into the same
# twenty phases. `source/studies/mjx/settings.py:check()` recomputes both sides
# at the start of EVERY trial and refuses to run if they have drifted apart,
# and the CLI additionally prints the EFFECTIVE budget of the process actually
# about to run, which is what catches a stray override. That per-sub-task
# budget -- 24,576,000 steps -- is repro2's own.
#
# NO ARM IS TOLD WHERE A PHASE BOUNDARY IS (CLAUDE.md (d)) -- and on this block
# there is nothing to tell: the ground and the observation never change. The
# phase grid still exists because the checkpoints sit on it, twenty per run,
# exactly where the continual block's do, so the two are compared point for
# point.
#
# USAGE
#   git pull                                    # needs the guards below to pass
#   DRY_RUN=1 bash scripts/train/queue_iclr_cheetah_noncontinual.sh
#   nohup bash scripts/train/queue_iclr_cheetah_noncontinual.sh \
#       > logs/cheetah_noncontinual.log 2>&1 &
#
#   # splitting the arms across two machines -- they write disjoint
#   # directories, so this is safe and so is re-running either side:
#   ARMS="ga es nes dns_gaussian" bash scripts/train/queue_iclr_cheetah_noncontinual.sh
#   ARMS="ppo trac redo cchain"   bash scripts/train/queue_iclr_cheetah_noncontinual.sh
#
#   # when done, send the tree home (the ant tree lives under the same root)
#   rsync -av --ignore-existing \
#       projects/iclr_2027/runs_mjx/ \
#       <home>:<repo>/projects/iclr_2027/runs_mjx/
#
# COST, measured on this body rather than estimated. A PPO job runs ~138
# updates a minute on an otherwise idle card, so 48,000 updates is ~6 h; the NE
# arms are ~1-1.5 h. That is ~240 job-hours for the four RL arms and ~50 for
# the four NE ones, so roughly 290 job-hours -- about 37 h on 8 cards at one
# job to a card. ONE JOB TO A CARD is the default: a 512 x 3 MJX rollout fills
# a GPU, which a 600 MiB gymnax job does not. NUM_TRIALS=5 halves the wall
# clock and is the thing to cut if time is short. Putting the four RL arms on
# their own machine roughly halves it too, and is the better cut.
#
# Safe to re-run and safe to overlap: run_condition skips any (method, cell,
# trial) whose training_metrics.json already exists.
#
# Env: GPUS, JOBS_PER_GPU (1), NUM_TRIALS (10), ROOT, ARMS, CELLS, DRY_RUN.
#
# DO NOT EDIT run_experiments.sh OR launch.sh IN PLACE WHILE THIS RUNS. bash
# reads a script incrementally, so an in-place rewrite under a live launcher
# corrupts its parse. Write a temp file and `mv` it.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ROOT="${ROOT:-projects/iclr_2027/runs_mjx}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
NUM_TRIALS="${NUM_TRIALS:-10}"
DRY_RUN="${DRY_RUN:-0}"
ARMS="${ARMS:-ga es nes dns_gaussian ppo trac redo cchain}"
CELLS="${CELLS:-cheetah}"
LOGS=logs/cheetah_noncontinual
mkdir -p "$LOGS"

# ------------------------------------------------------------ version guard
# The same check whose absence cost eight GA trials on 2026-09-09: a launcher
# started against a checkout whose trainer could not yet write the centroid,
# and Python reads its source at process START, so runs that FINISHED after
# the fix still had no centroid in them. Nothing recovers it post hoc.
if [ ! -f source/studies/mjx/cli.py ]; then
    echo "FATAL: this checkout has no mjx study. Run: git pull" >&2
    exit 1
fi
for f in train_nes train_ppo; do
    if ! grep -q 'save_checkpoints' "source/studies/generalists/${f}.py"; then
        echo "FATAL: source/studies/generalists/${f}.py cannot write" >&2
        echo "       checkpoints.npz, so no centroid figure is possible." >&2
        echo "       This checkout is too old. Run: git pull" >&2
        exit 1
    fi
done
# The cheetah cells AND its friction grid. Without the grid the friction cells
# raise KeyError on f['low'], and a checkout with the cells but not the grid
# would run the noise family and die on the other one later.
if ! .venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from source.studies.mjx import settings as S
assert 'cheetah' in S.CELLS and 'cheetah_friction' in S.CELLS
from source.envs import mjx
assert 'friction' in mjx.ENV_CONFIGS['CheetahRun']
" 2>/dev/null; then
    echo "FATAL: the cheetah cells or its friction grid are missing." >&2
    echo "       This checkout is too old. Run: git pull" >&2
    exit 1
fi
if ! grep -q 'cheetah_noncontinual)' scripts/train/run_experiments.sh; then
    echo "FATAL: run_experiments.sh has no cheetah blocks. Run: git pull" >&2
    exit 1
fi

# A real import of the settings, which is also the compute-match assertion.
# Cheaper to fail here than 80 times in a row inside the launcher. It also
# prints the effective budget, so the line to read before walking away is
# "this run : 4.915e+08 steps over 20 phases (100.0% of matched)".
for cell in $CELLS; do
    if ! .venv/bin/python source/studies/mjx/cli.py \
            --env "$cell" --method ppo --output_dir /tmp/cheetah_preflight \
            --dry_run; then
        echo "FATAL: preflight failed on cell $cell." >&2
        exit 1
    fi
done
echo "guards OK"

echo "root       : $ROOT/mjx/noncontinual"
echo "cells      : $CELLS"
echo "arms       : $ARMS"
echo "trials     : 1..$NUM_TRIALS"
echo "jobs       : $(( $(echo $ARMS | wc -w) * $(echo $CELLS | wc -w) * NUM_TRIALS ))"

# ------------------------------------------------------------ GPU pool
# One token per entry, leased by launch.sh -- repeat a GPU to oversubscribe it.
if [ -z "${GPUS:-}" ]; then
    VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')
    VISIBLE=${VISIBLE:-0}
    GPUS=""
    for g in $VISIBLE; do
        for _ in $(seq 1 "$JOBS_PER_GPU"); do GPUS="$GPUS $g"; done
    done
fi
echo "gpus       : $GPUS"

if [ "$DRY_RUN" != "0" ]; then
    DRY_RUN=1 GPUS="$GPUS" NUM_TRIALS="$NUM_TRIALS" ENVS="$CELLS" \
      PROJECT_ROOT="$ROOT" LOG_DIR="$LOGS" \
      bash scripts/train/launch.sh cheetah_noncontinual $ARMS
    echo "=== DRY RUN -- nothing queued ==="
    exit 0
fi

echo "=== $(date -Is) starting ==="
GPUS="$GPUS" NUM_TRIALS="$NUM_TRIALS" ENVS="$CELLS" \
  PROJECT_ROOT="$ROOT" LOG_DIR="$LOGS" \
  bash scripts/train/launch.sh cheetah_noncontinual $ARMS
code=$?
echo "=== $(date -Is) done (exit=$code) ==="
echo "failures:"
grep -v 'exit=0' "$LOGS/status.tsv" 2>/dev/null || echo "  none"
