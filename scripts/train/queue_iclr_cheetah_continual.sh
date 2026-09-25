#!/bin/bash
# ============================================================================
# THE CHEETAH CONTINUAL BLOCK, both families. Run AFTER the stationary one.
#
# 160 trials: 8 arms x 2 cells x 10 trials, into projects/iclr_2027/runs_mjx.
#
# THE GRID
#
#     cells   cheetah_noise      a sub-task is a fixed offset added to the
#                                body's 17 observations, sigma 2.0. The BODY is
#                                unchanged.
#             cheetah_friction   a sub-task is a ground-friction multiplier
#                                drawn log-uniform in [0.05, 5.0] per trial.
#                                The PHYSICS changes and the observation does
#                                not.
#     arms    ga es nes dns_gaussian ppo trac redo cchain
#     trials  1..10
#
# TEN SUB-TASKS OVER TWENTY PHASES, so each is visited exactly TWICE and the
# second visit is the retention measurement. runs_repro2/mujoco ran TWELVE
# sub-tasks at `task_period 0` -- nothing was ever revisited, so forgetting was
# not measurable there at all, only relearning speed. That is the column this
# block exists to fill, and it is the gymnax grid's structure
# (`NUM_TASKS=20 TASK_PERIOD=10`), which is what lets the bodies be read side
# by side.
#
# THE TWO CELLS ARE THE SAME EXPERIMENT WITH ONE THING CHANGED. Same body, same
# reward, same policy, same budget, same phase grid, same schedule; the ONLY
# difference is what a sub-task vector means (`task_mod` in
# source/envs/mjx.py). So the pair separates "the optimum moved because the
# policy's input moved" from "the policy reads the same numbers and the right
# response to them is different" -- a comparison the gymnax families cannot
# make, since nothing there changes the physics.
#
# ---------------------------------------------------------------------------
# WHAT THIS BLOCK IS FOR, AND HOW IT CAN GO WRONG
#
# The claim being tested is that the RL arms lose plasticity here and the NE
# arms do not. runs_repro2/mujoco is where that showed up under observation
# noise, and it did NOT show up under friction -- so the honest expectation is
# ONE of these two cells reproduces it and the other may not. Both are run
# because "friction does not hurt PPO" is a result about the mujoco_playground
# cheetah, and this is a different body on a different reward; the probe found
# a solved policy dropping from 4531 to 706 zero-shot across friction
# multipliers, which is a much sharper shift than that tree saw.
#
# THE FAILURE MODE TO WATCH IS PPO NOT FAILING FOR AN UNINTERESTING REASON.
# In the probe, PPO reached ~4600 of a ~5000 ceiling within 1800 of its 48,000
# updates -- 4% of the budget, from scratch. A phase here is 2400 updates. So
# on any sub-task PPO can simply re-solve from scratch inside its phase, and a
# flat, healthy-looking PPO curve would then mean "the phase budget is generous"
# rather than "PPO retains". That is exactly the trap block_brax_continual hit
# on the ant, where brax's tuned rollout let the continual arm END SUB-TASKS
# ABOVE its own stationary control (4282 against 4380) -- a continual arm that
# beats its control is not measuring plasticity loss.
#
# Read these three things before believing a null result:
#
#   1. per-phase end score vs the stationary control. If the continual arm ends
#      phases at or above `mjx/noncontinual`, the sequence is not a shift for
#      it and nothing here is about plasticity.
#   2. the ZERO-SHOT score at each boundary -- what the carried policy gets on
#      the new sub-task before any update. If that is already high, the
#      sub-tasks are not conflicting and the arm is a generalist, not a
#      retainer.
#   3. the SECOND visit against the first. This is the whole point of the
#      revisit and the thing repro2 could not measure.
#
# IF PPO DOES NOT DEGRADE, THE FIRST KNOB IS THE ROLLOUT SHAPE, NOT THE TASK.
# repro2's cheetah PPO used an 8x bigger update at a 3x smaller learning rate
# (4096 x 20 = 81,920 env steps an update at lr 1e-4, against `_MJX_PPO`'s
# 512 x 20 = 10,240 at 3e-4), which is the small-signal, repeatedly-reused-
# gradient regime the loss-of-plasticity literature measures in. The stationary
# probe cannot choose between the shapes -- both saturate -- so the choice was
# deferred to here. Run the contrast at the SAME total budget with:
#
#     ARMS="ppo trac redo cchain" ROOT=projects/iclr_2027/runs_mjx_bigbatch \
#     MJX_EXTRA="--num_updates 6000 --task_interval 300 --ppo_override \
#       num_envs=4096 num_steps=20 num_epochs=8 learning_rate=0.0001 \
#       reward_scale=0.1" \
#     bash scripts/train/queue_iclr_cheetah_continual.sh
#
# 6000 x 81,920 = 4.9152e8, the identical wall of environment steps over the
# identical twenty phases, so the two differ in shape and in nothing else. A
# SEPARATE ROOT is not optional: same cell, same arm name, different settings,
# and run_condition would otherwise skip the second as "already done".
#
# DO NOT reach for a harder task first. Widening the perturbation until PPO
# breaks is how a benchmark stops measuring plasticity and starts measuring
# whether anything can learn at all -- the gymnax MountainCar sigma story.
# ---------------------------------------------------------------------------
#
# ONE KNOWN CONFOUND ON THE NOISE CELL, LEFT ON DELIBERATELY. PPO normalises
# observations (`normalize_obs` in PPO_CONFIGS) and the NE arms have no
# normaliser at all. brax's running statistics are part of the preserved
# training state, so the sample count accumulates across sub-tasks and updates
# decay as 1/count: each sub-task's fixed offset is POOLED into one estimate
# rather than tracked, and the network is handed the current offset uncorrected
# AND a signal divided by the pooled spread. So part of any RL-vs-NE gap on
# `cheetah_noise` is this, and since that gap is the headline claim it matters
# here more than anywhere. It is left on because it is what runs_repro2 did and
# turning it off would make these numbers incomparable with that tree; the
# CONTROL is a second pass with
#
#     ARMS="ppo trac redo cchain" CELLS=cheetah_noise \
#     ROOT=projects/iclr_2027/runs_mjx_nonorm \
#     MJX_EXTRA="--ppo_override normalize_obs=0" \
#     bash scripts/train/queue_iclr_cheetah_continual.sh
#
# Do NOT "fix" it by resetting the normaliser per sub-task instead: the
# perturbation is a constant additive offset, which is exactly what
# mean-subtraction removes, so a freshly fitted normaliser cancels the
# manipulation and reports a retention number that means nothing. The friction
# cell is untouched by any of this -- its observation does not move.
#
# NO ARM IS TOLD WHERE A BOUNDARY IS (CLAUDE.md (d)). C-CHAIN's
# `cchain_reset_on_switch` is off, the GA and DNS re-evaluate their stored
# population EVERY generation rather than only at a transition, and nothing is
# passed a phase index. `--oracle` on the CLI is the deliberately-informed
# control arm and is not used here.
#
# COMPUTE-MATCHED: 4.9152e8 environment steps for every arm, as 320 x 512 x 3 x
# 1000 for the NE arms and 48,000 x 512 x 20 for the RL ones, cut into the same
# twenty phases (16 generations against 2400 updates). settings.check()
# recomputes both sides at the start of EVERY trial, and the CLI prints the
# EFFECTIVE budget of the process about to run -- which is what catches a stray
# override, including the two contrasts above. That per-sub-task budget,
# 24,576,000 steps, is repro2's own.
#
# RUN THE STATIONARY BLOCK FIRST. Point 1 above is unanswerable without it.
# queue_iclr_cheetah_noncontinual.sh, same root.
#
# USAGE
#   git pull
#   DRY_RUN=1 bash scripts/train/queue_iclr_cheetah_continual.sh
#   nohup bash scripts/train/queue_iclr_cheetah_continual.sh \
#       > logs/cheetah_continual.log 2>&1 &
#
#   # one family at a time, or one family per machine:
#   CELLS=cheetah_noise    bash scripts/train/queue_iclr_cheetah_continual.sh
#   CELLS=cheetah_friction bash scripts/train/queue_iclr_cheetah_continual.sh
#
#   # splitting the arms across machines (disjoint directories, so safe):
#   ARMS="ga es nes dns_gaussian" bash scripts/train/queue_iclr_cheetah_continual.sh
#   ARMS="ppo trac redo cchain"   bash scripts/train/queue_iclr_cheetah_continual.sh
#
#   # when done, send the tree home (the ant tree shares this root)
#   rsync -av --ignore-existing \
#       projects/iclr_2027/runs_mjx/ \
#       <home>:<repo>/projects/iclr_2027/runs_mjx/
#
# COST, measured on this body. ~6 h a PPO-family trial (138 updates a minute on
# an idle card) and ~1-1.5 h an NE trial, one job to a card. 160 trials is
# ~580 job-hours: about 73 h on 8 cards. THE CELLS ARE INDEPENDENT, so the
# obvious split is one family per machine (~36 h each), and the next cut is
# NUM_TRIALS=5.
#
# Safe to re-run and safe to overlap: run_condition skips any (method, cell,
# trial) whose training_metrics.json already exists.
#
# Env: GPUS, JOBS_PER_GPU (1), NUM_TRIALS (10), ROOT, ARMS, CELLS, MJX_EXTRA,
# DRY_RUN.
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
CELLS="${CELLS:-cheetah_noise cheetah_friction}"
LOGS=logs/cheetah_continual
mkdir -p "$LOGS"

# ------------------------------------------------------------ version guard
# Python reads its source at process START, so a launcher begun against a
# checkout that cannot save the centroid writes runs with no centroid in them
# even if the fix lands while it runs, and nothing recovers it post hoc.
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
if ! .venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from source.studies.mjx import settings as S
assert 'cheetah_noise' in S.CELLS and 'cheetah_friction' in S.CELLS
from source.envs import mjx
assert 'friction' in mjx.ENV_CONFIGS['CheetahRun']
" 2>/dev/null; then
    echo "FATAL: the cheetah cells or its friction grid are missing." >&2
    echo "       This checkout is too old. Run: git pull" >&2
    exit 1
fi
if ! grep -q 'cheetah_continual)' scripts/train/run_experiments.sh; then
    echo "FATAL: run_experiments.sh has no cheetah blocks. Run: git pull" >&2
    exit 1
fi

# A real import of the settings on BOTH cells -- they resolve different task
# options and a broken one would otherwise only surface 80 jobs in. Read the
# "this run" line it prints: with MJX_EXTRA set, that is the only place the
# effective budget of a contrast run is visible before it starts.
for cell in $CELLS; do
    if ! .venv/bin/python source/studies/mjx/cli.py \
            --env "$cell" --method ppo --output_dir /tmp/cheetah_preflight \
            --dry_run ${MJX_EXTRA:-}; then
        echo "FATAL: preflight failed on cell $cell." >&2
        exit 1
    fi
done

# The stationary control. Not required to TRAIN -- only the analysis reads it --
# but point 1 in the header (does the continual arm end phases above its own
# control?) is the first thing to check and is unanswerable without it.
if [ ! -d "$ROOT/mjx/noncontinual/ppo/cheetah" ]; then
    echo "NOTE: $ROOT/mjx/noncontinual/ppo/cheetah does not exist. The"
    echo "      stationary control is what says whether these curves are"
    echo "      degrading at all; run queue_iclr_cheetah_noncontinual.sh"
    echo "      first, or bring that tree over before the analysis."
fi
echo "guards OK"

echo "root       : $ROOT/mjx/continual"
echo "cells      : $CELLS"
echo "arms       : $ARMS"
echo "trials     : 1..$NUM_TRIALS"
echo "phases     : 20 over 10 sub-tasks (each visited twice)"
[ -n "${MJX_EXTRA:-}" ] && echo "extra      : ${MJX_EXTRA} (CONTRAST RUN -- check ROOT is not the reported one)"
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
      PROJECT_ROOT="$ROOT" LOG_DIR="$LOGS" MJX_EXTRA="${MJX_EXTRA:-}" \
      bash scripts/train/launch.sh cheetah_continual $ARMS
    echo "=== DRY RUN -- nothing queued ==="
    exit 0
fi

echo "=== $(date -Is) starting ==="
GPUS="$GPUS" NUM_TRIALS="$NUM_TRIALS" ENVS="$CELLS" \
  PROJECT_ROOT="$ROOT" LOG_DIR="$LOGS" MJX_EXTRA="${MJX_EXTRA:-}" \
  bash scripts/train/launch.sh cheetah_continual $ARMS
code=$?
echo "=== $(date -Is) done (exit=$code) ==="
echo "failures:"
grep -v 'exit=0' "$LOGS/status.tsv" 2>/dev/null || echo "  none"
