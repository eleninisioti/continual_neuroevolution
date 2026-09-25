#!/bin/bash
# ============================================================================
# THE ANT CONTINUAL BLOCK, both families. Run AFTER the stationary one.
#
# 160 trials: 8 arms x 2 cells x 10 trials, into projects/iclr_2027/runs_mjx.
#
# THE GRID
#
#     cells   ant_noise      a sub-task is a fixed offset added to the ant's 27
#                            observations, sigma 2.0. The BODY is unchanged.
#             ant_friction   a sub-task is a ground-friction multiplier drawn
#                            log-uniform in [0.05, 5.0] per trial. The PHYSICS
#                            changes and the observation does not.
#     arms    ga es nes dns_gaussian ppo trac redo cchain
#     trials  1..10
#
# TEN SUB-TASKS OVER TWENTY PHASES, so each is visited exactly TWICE and the
# second visit is the retention measurement. That is the gymnax grid's
# structure (`NUM_TASKS=20 TASK_PERIOD=10`), which is the whole reason it is
# this and not something else: the two bodies have to be readable side by side.
# A sequence with no revisit measures how fast an arm relearns and cannot
# measure what it kept.
#
# THE TWO CELLS ARE THE SAME EXPERIMENT WITH ONE THING CHANGED. Same body, same
# reward (speed tracking at 2.0 m/s), same policy, same budget, same phase
# grid, same schedule; the ONLY difference is what a sub-task vector means --
# `task_mod` in source/envs/mjx.py. So the pair separates "the optimum moved
# because the policy's input moved" from "the policy reads the same numbers and
# the right response to them is different", which is the one comparison the
# gymnax families cannot make (nothing there changes the physics) and the one
# runs_repro2's ant trees cannot make either (their obsnoise tree turned out to
# be a friction experiment -- see block_mujoco_continual's note).
#
# THE REWARD IS THE PAPER'S CONTINUAL ANT REWARD, not brax's forward velocity.
# The paper measured that under the stock unbounded-velocity reward an ant
# re-routes around a friction change -- four legs, many gaits -- so friction
# alone was not a shift at all. Speed tracking is what makes the friction cell
# a benchmark. It is on for BOTH cells so the two stay comparable, and for the
# stationary control too.
#
# WHY THE FRICTION DRAW IS RANDOM AND NOT THE THREE-VALUE CYCLE. The paper's
# `friction_cycle` has three values (x1.0, x0.2, x5.0), which does not divide
# twenty phases evenly and would give one of them six visits against seven --
# a different number of revisits per sub-task, which is exactly the confound
# `pairwise`/`sampled` were added to break. The log-uniform draw is the
# Slippery-Ant protocol and is what runs_repro2's own ant block used
# (`ANT_FRICTION_LOW_MULT=0.05`). Sub-task 0 is pinned to x1.0, so it is the
# stationary experiment exactly as under the noise family. A trial is its own
# ten multipliers, seeded off the TRIAL index alone, so every method at a given
# trial faces the identical sequence at the identical steps -- CLAUDE.md (c).
#
# NO ARM IS TOLD WHERE A BOUNDARY IS (CLAUDE.md (d)). C-CHAIN's
# `cchain_reset_on_switch` is off, the GA and DNS re-evaluate their stored
# population EVERY generation rather than only at a transition, and nothing is
# passed a phase index. `--oracle` on the CLI is the deliberately-informed
# control arm and is not used here.
#
# COMPUTE-MATCHED: 4.9152e8 environment steps for every arm, as 320 x 512 x 3 x
# 1000 for the NE arms and 48,000 x 512 x 20 for the RL ones, cut into the same
# twenty phases (16 generations against 2400 updates).
# `source/studies/mjx/settings.py:check()` recomputes both sides at the start of
# EVERY trial and refuses to run if they have drifted apart. DNS is matched by
# `refresh=True`: it re-scores its 256-member repertoire AND rolls out 256
# offspring each generation, which is the 512 the other NE arms spend -- the
# one place the old ant block got it wrong by default.
#
# ONE KNOWN CONFOUND ON THE NOISE CELL, LEFT ON DELIBERATELY. PPO normalises
# observations (`normalize_obs` in PPO_CONFIGS['ant']) and the NE arms have no
# normaliser at all. brax's running statistics are part of the preserved
# training state, so the sample count accumulates across sub-tasks and updates
# decay as 1/count: each sub-task's fixed offset is POOLED into one estimate
# rather than tracked, and the network is handed the current offset uncorrected
# AND a signal divided by the pooled spread. Measured on the old ant tree over
# 27 dims, the mean stayed put while the variance went 2.06 -> 5.60. So part of
# any RL-vs-NE gap on `ant_noise` is this. It is left on because it is what the
# paper's own ant runs did and turning it off would make these numbers
# incomparable with them; the CONTROL is a second pass with
#
#     ARMS="ppo trac redo cchain" CELLS=ant_noise ROOT=<a different root> \
#     ANT_EXTRA="--ppo_override normalize_obs=0" bash scripts/train/queue_iclr_ant_continual.sh
#
# Do NOT "fix" it by resetting the normaliser per sub-task instead: the
# perturbation is a constant additive offset, which is exactly what
# mean-subtraction removes, so a freshly fitted normaliser cancels the
# manipulation and reports a retention number that means nothing. The friction
# cell is untouched by any of this -- its observation does not move.
#
# RUN THE STATIONARY BLOCK FIRST. Everything reported here is read against it,
# and if an arm cannot walk on the healthy ant then its continual curve says
# nothing about task changes. queue_iclr_ant_noncontinual.sh, same root.
#
# USAGE
#   git pull                                    # needs the guards below to pass
#   DRY_RUN=1 bash scripts/train/queue_iclr_ant_continual.sh
#   nohup bash scripts/train/queue_iclr_ant_continual.sh \
#       > logs/ant_continual.log 2>&1 &
#
#   # one family at a time, or one family per machine:
#   CELLS=ant_friction bash scripts/train/queue_iclr_ant_continual.sh
#
#   # splitting the arms across two machines (they write disjoint directories,
#   # so this is safe and so is re-running either side):
#   ARMS="ga es nes dns_gaussian" bash scripts/train/queue_iclr_ant_continual.sh
#   ARMS="ppo trac redo cchain"   bash scripts/train/queue_iclr_ant_continual.sh
#
#   # when done, send the tree home
#   rsync -av --ignore-existing \
#       projects/iclr_2027/runs_mjx/ \
#       <home>:<repo>/projects/iclr_2027/runs_mjx/
#
# COST. ~1 h a trial for the NE arms and ~3 h for the RL ones at this budget,
# ONE JOB TO A CARD -- 512 x 3 MJX ant rollouts fill a GPU. 160 trials is about
# 320 job-hours, so roughly 40 h on 8 cards, on top of the stationary block's
# 20 h. JOBS_PER_GPU is 1 by default; raise it only once `nvidia-smi` has shown
# you what one job costs on your cards. NUM_TRIALS=5 halves it.
#
# Safe to re-run and safe to overlap: run_condition skips any
# (method, cell, trial) whose training_metrics.json already exists.
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
CELLS="${CELLS:-ant_noise ant_friction}"
LOGS=logs/ant_continual
mkdir -p "$LOGS"

# ------------------------------------------------------------ version guard
# See queue_iclr_ant_noncontinual.sh: Python reads its source at process START,
# so a launcher begun against a checkout that cannot save the centroid writes
# runs with no centroid in them even if the fix lands while it runs, and
# nothing recovers it post hoc.
if [ ! -f source/studies/mjx/cli.py ]; then
    echo "FATAL: this checkout has no ant study. Run: git pull" >&2
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
if ! grep -q 'def friction_multipliers' source/envs/mjx.py; then
    echo "FATAL: source/envs/mjx.py has no friction sub-tasks. Run: git pull" >&2
    exit 1
fi
if ! grep -q 'ant_continual)' scripts/train/run_experiments.sh; then
    echo "FATAL: run_experiments.sh has no ant blocks. Run: git pull" >&2
    exit 1
fi

# A real import of the settings, which is also the compute-match assertion, on
# BOTH cells -- they resolve different task options and a broken one would
# otherwise only surface 80 jobs in.
for cell in $CELLS; do
    if ! .venv/bin/python source/studies/mjx/cli.py \
            --env "$cell" --method ppo --output_dir /tmp/ant_preflight \
            --dry_run; then
        echo "FATAL: preflight failed on cell $cell." >&2
        exit 1
    fi
done

# The stationary control. Not required to TRAIN -- only the analysis reads it --
# but running the continual block first is how a sweep ends up with curves and
# nothing to read them against.
if [ ! -d "$ROOT/mjx/noncontinual" ]; then
    echo "NOTE: $ROOT/mjx/noncontinual does not exist. The stationary control"
    echo "      is what every number here is read against; run"
    echo "      scripts/train/queue_iclr_ant_noncontinual.sh first, or bring"
    echo "      that tree over before the analysis."
fi
echo "guards OK"

echo "root       : $ROOT/mjx/continual"
echo "cells      : $CELLS"
echo "arms       : $ARMS"
echo "trials     : 1..$NUM_TRIALS"
echo "phases     : 20 over 10 sub-tasks (each visited twice)"
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
      bash scripts/train/launch.sh ant_continual $ARMS
    echo "=== DRY RUN -- nothing queued ==="
    exit 0
fi

echo "=== $(date -Is) starting ==="
GPUS="$GPUS" NUM_TRIALS="$NUM_TRIALS" ENVS="$CELLS" \
  PROJECT_ROOT="$ROOT" LOG_DIR="$LOGS" \
  bash scripts/train/launch.sh ant_continual $ARMS
code=$?
echo "=== $(date -Is) done (exit=$code) ==="
echo "failures:"
grep -v 'exit=0' "$LOGS/status.tsv" 2>/dev/null || echo "  none"
