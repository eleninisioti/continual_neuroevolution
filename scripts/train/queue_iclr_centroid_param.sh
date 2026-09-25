#!/bin/bash
# ============================================================================
# The PHYSICS half of the ICLR 2027 gymnax experiment, into
# projects/iclr_2027/runs_param.
#
# Same eight arms, same budget, same 20 sub-tasks at period 10 as
# queue_iclr_centroid.sh. ONE thing differs: what a sub-task IS.
#
#   runs_centroid   a sub-task adds a fixed vector to the OBSERVATION.
#   runs_param      a sub-task rescales the BODY and leaves the observation
#                   alone -- CartPole's pole length, Acrobot's link masses,
#                   MountainCar's gravity.
#
# Sub-task 0 is the stock body under both, so the noncontinual block is
# literally the same experiment and is SYMLINKED from runs_centroid rather
# than trained a second time (see the symlink below).
#
# WHICH KNOB AND HOW FAR ARE MEASURED, NOT GUESSED. The multiplier is drawn
# log-uniformly over a per-env range set in source/utils/task_sequence.py from
# the zero-shot sweep in source/envs/gymnax_classic.py: a rescaling is only a
# sub-task if a specialist trained on the stock body FAILS it, and only usable
# if a specialist trained on it SUCCEEDS. That fixes CartPole at 0.5x-2x on
# `length`, Acrobot at 1/1.15x-1.15x on `mass` and MountainCar at 1/1.5x-1.5x
# on `gravity`.
#
# FOUR BUGS WERE FIXED TO MAKE THIS RUNNABLE (2026-09-08). Queued before them,
# this sweep would have produced a tree that looked complete and was not
# comparable:
#
#   1. ES/NES had NO param branch. They ignored --task_type (the runner did not
#      even pass it) and would have run the OBSERVATION-NOISE experiment into
#      this tree while the other six arms ran the physics one.
#   2. GA/DNS and RL carried DIFFERENT per-trainer parameter tables --
#      CartPole [0.98, 98.0] against [0.098, 198.0], MountainCar [0.000833,
#      0.0075] against [0.00125, 0.005]. Both drew from the same trial-seeded
#      key, so the NE and RL arms of one compute-matched comparison faced
#      different sub-task sequences (CLAUDE.md rule c). There is now one table.
#   3. The param branch never called cycle_task_sequence, so --task_period was
#      silently ignored: 20 DISTINCT sub-tasks instead of ten seen twice, and
#      nothing revisited means forgetting cannot be measured at all.
#   4. Every trainer emitted checkpoints.npz and results.json only under
#      `noise`, so a param run saved nothing for evaluate_continual.py and was
#      invisible to verify_runs.py's budget check. Both now carry the
#      multiplier sequence and the evaluator rebuilds each body.
#
# ABOUT THE `_sigma1.0` IN THE DIRECTORY NAMES. It is a NAME, not a setting:
# the observation is untouched here and no offset is drawn. The suffix is kept
# because every analysis script recovers the environment by splitting the cell
# name on `_sigma`, so this tree reads with the same `--sigma 1.0` the noise
# tree does. What the runs actually are is recorded in each run's config as
# `task_type: param`; the tree root is what tells the two experiments apart.
#
# STARTS ONLY WHEN THE NOISE SWEEP IS DONE. The box is full: 32 concurrent
# jobs over 8 GPUs at 72-100% utilisation. Starting now would not add
# throughput, it would halve the rate of a sweep that is already half done.
# Touch the STOP file below to cancel before it launches.
#
# DO NOT EDIT scripts/train/run_experiments.sh OR launch.sh WHILE THIS RUNS,
# unless the edit lands as an ATOMIC RENAME (write a temp file, then mv). bash
# reads a script incrementally, so an in-place rewrite under a live launcher
# corrupts its parse -- that happened on 2026-09-08 and cost an hour of jobs
# exiting 2 while the launcher reported itself as running.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ARMS="ga_isoline ga dns es nes ppo trac redo cchain"
ROOT=projects/iclr_2027/runs_param
TRIALS="${NUM_TRIALS:-10}"
STOP=projects/iclr_2027/runs_param/STOP

mkdir -p "$ROOT/gymnax"

# ---------------------------------------------------------------- the wait
# Poll rather than `wait`: the two launchers were started from other shells, so
# this process is not their parent and cannot wait(2) on them.
echo "=== $(date -Is) waiting for the noise sweep to finish ==="
while true; do
    if [ -f "$STOP" ]; then
        echo "=== $(date -Is) STOP file present, exiting without launching ==="
        exit 0
    fi
    running=$(pgrep -fc 'scripts/train/(queue_iclr_centroid(_ga)?|launch)\.sh' || true)
    [ "${running:-0}" -eq 0 ] && break
    sleep 120
done
# The launchers exit before their last trainer has flushed its final artifacts.
sleep 120
echo "=== $(date -Is) noise sweep done, starting the physics sweep ==="

# ------------------------------------------------------- noncontinual block
# NOT retrained: sub-task 0 of a param run IS the stock body, so the
# noncontinual phase of this experiment is the noncontinual phase of the noise
# one -- same trainer, same seeds, same environment, 240 jobs saved. A symlink
# rather than a copy so there is exactly one set of files to be right.
if [ ! -e "$ROOT/gymnax/noncontinual" ]; then
    ln -s ../../runs_centroid/gymnax/noncontinual "$ROOT/gymnax/noncontinual"
    echo "linked noncontinual -> runs_centroid/gymnax/noncontinual"
fi

# ---------------------------------------------------------- continual block
# FOUR JOBS PER GPU, as in queue_iclr_centroid.sh: launch.sh leases one token
# per entry in GPUS, so repeating an index is how it is told to oversubscribe.
GPUS_DEFAULT=""
for g in 0 1 2 3 4 5 6 7; do for _ in 1 2 3 4; do GPUS_DEFAULT="$GPUS_DEFAULT $g"; done; done
export GPUS="${GPUS:-$GPUS_DEFAULT}"

NUM_TRIALS="$TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  TASK_TYPE=param NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 SIGMAS=1.0 \
  LOG_DIR=logs/launch_param_continual \
  bash scripts/train/launch.sh gymnax_continual $ARMS

echo "=== $(date -Is) physics sweep done ==="
echo "failures:"
grep -hv 'exit=0' logs/launch_param_continual/status.tsv 2>/dev/null || echo "  none"
