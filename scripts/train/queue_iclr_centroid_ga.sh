#!/bin/bash
# ============================================================================
# The `ga` arm of the ICLR 2027 gymnax re-run, into the SAME tree as
# scripts/train/queue_iclr_centroid.sh: projects/iclr_2027/runs_centroid.
#
# WHY THIS IS A SEPARATE SCRIPT. queue_iclr_centroid.sh deliberately queued
# only `ga_isoline`, on the reasoning that a DNS-over-GA gap at a confounded
# operator is a claim about two things at once. That reasoning justifies ADDING
# ga_isoline; it does not justify dropping `ga`, which is the GA as published
# (Such et al., 2017) and the arm every GA number in the literature refers to.
# With both, the gymnax grid carries the full 2x2 minus one cell:
#
#                         gaussian     isoline
#   fitness truncation    ga           ga_isoline
#   dominated novelty     --           dns
#
# so `ga` vs `ga_isoline` isolates the OPERATOR at fixed selection, and `dns`
# vs `ga_isoline` isolates the SELECTION at fixed operator.
#
# It is a separate file because the other queue was already running when this
# was written, and bash reads a script incrementally -- editing a live one is
# how an earlier launcher spent an hour exiting 2 on a syntax error while
# reporting itself as running. For the same reason this script does not touch
# launch.sh or run_experiments.sh, and only reads them.
#
# Every setting below is copied from queue_iclr_centroid.sh so the arms are
# comparable: same PROJECT_ROOT, same 10 trials from BASE_SEED 42, same
# TRACK_DIVERSITY, same 20 sub-tasks at period 10 and 200 generations each,
# same sigma 1.0. `ga` needs no operator flags -- gaussian mutation at the
# trainer's own reference mutation_std is the published method.
#
# SAFE TO RE-RUN: run_condition skips a (method, env, trial) whose output
# already exists, so an interrupted queue resumes where it stopped.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ARMS="ga"

# EIGHT SLOTS, NOT THIRTY-TWO. The main queue leases 4 per GPU and is still
# running; this one takes one per GPU on top of it, so it adds ~25% load rather
# than doubling it. That is deliberate: `ga` is ~60 jobs against that queue's
# ~480, and finishing it alongside costs the other arms a little wall-clock
# instead of costing this arm a full day of waiting.
#
# The contention caveat in launch.sh's header is about RL: concurrent PPO jobs
# are not bit-reproducible because XLA autotunes different kernels under memory
# pressure. It does not apply here -- GA and ES were bit-identical over 600
# generations x 30 trials at that concurrency. Run this alone (GPUS="0 1 ... 7"
# with the main queue stopped) if that ever stops being true.
GPUS_DEFAULT="0 1 2 3 4 5 6 7"
export GPUS="${GPUS:-$GPUS_DEFAULT}"
ROOT=projects/iclr_2027/runs_centroid
TRIALS="${NUM_TRIALS:-10}"

echo "=== $(date -Is) ga noncontinual block ==="
NUM_TRIALS="$TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  LOG_DIR=logs/launch_centroid_ga_noncontinual \
  bash scripts/train/launch.sh gymnax_noncontinual $ARMS

echo "=== $(date -Is) ga continual block ==="
NUM_TRIALS="$TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 SIGMAS=1.0 \
  LOG_DIR=logs/launch_centroid_ga_continual \
  bash scripts/train/launch.sh gymnax_continual $ARMS

echo "=== $(date -Is) ga blocks done ==="
echo "failures:"
grep -hv 'exit=0' logs/launch_centroid_ga_*/status.tsv 2>/dev/null || echo "  none"
