#!/bin/bash
# ============================================================================
# The ICLR 2027 gymnax re-run, into projects/iclr_2027/runs_centroid.
#
# What is different from the tree in projects/iclr_2027/runs:
#
#   1. THE REPORTED CURVE IS A SEPARATE EVALUATION. Every NE trainer now scores
#      `centroid_fitness` and `elite_eval_fitness` on FRESH keys over
#      --report_episodes (10) episodes, the same protocol and episode count the
#      RL trainers' `mean_reward` has always used. The old figure plotted
#      `best_fitness`, which is a max over 512 individuals of a 3-episode mean
#      whose FIRST draw did the selecting -- optimistically biased, and by a
#      different amount per method (+5 for the GA, +53 for DNS), so the bias
#      did not cancel in the comparison.
#   2. THE GA IS OPERATOR-MATCHED TO DNS. `ga_isoline` breeds with DNS's
#      Iso+LineDD at the same 0.05 / 0.5 widths and selects by fitness
#      truncation, so `dns` vs `ga_isoline` differs in the SELECTION RULE
#      alone. The gaussian-mutation `ga` is not queued: with the operator
#      confounded, a DNS-over-GA gap was a claim about two things at once.
#
# Everything else matches the reported tree: 20 sub-tasks at period 10 (each
# seen twice), 200 generations each, 10 trials, sigma 1.0, and the same
# 3.072e9-environment-step budget for every method.
#
# The noncontinual block runs FIRST. It is ~7x cheaper, FT subtracts it from
# the continual curve so it is needed either way, and running it first means a
# configuration error costs six hours rather than two days.
#
# DO NOT EDIT scripts/train/run_experiments.sh OR launch.sh WHILE THIS RUNS.
# bash reads a script incrementally: on 2026-09-08 an edit landed under a live
# launcher and it spent the next hour exiting 2 on `syntax error near
# unexpected token ;;` while reporting itself as running.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ARMS="ga_isoline dns es nes ppo trac redo cchain"

# FOUR JOBS PER GPU. launch.sh leases one token per entry in GPUS, so repeating
# an index is how it is told to oversubscribe. Measured on this box: a gymnax
# trainer holds ~600 MiB of a 49 GB card and ~1.7 cores of 256, so one job per
# GPU left the machine almost idle and the wall-clock was 8x longer than the
# hardware needed.
#
# The tradeoff, from launch.sh's own header: concurrent RL jobs are NOT
# bit-reproducible against a run made alone -- under memory pressure XLA
# autotunes different kernels and the reduction order changes. The NE methods
# were unaffected at 30/30 trials. That is accepted here: these runs are read
# as 10-trial distributions, not reproduced update for update.
GPUS_DEFAULT=""
for g in 0 1 2 3 4 5 6 7; do for _ in 1 2 3 4; do GPUS_DEFAULT="$GPUS_DEFAULT $g"; done; done
export GPUS="${GPUS:-$GPUS_DEFAULT}"
ROOT=projects/iclr_2027/runs_centroid
TRIALS="${NUM_TRIALS:-10}"

echo "=== $(date -Is) noncontinual block ==="
NUM_TRIALS="$TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  LOG_DIR=logs/launch_centroid_noncontinual \
  bash scripts/train/launch.sh gymnax_noncontinual $ARMS

echo "=== $(date -Is) continual block ==="
NUM_TRIALS="$TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 SIGMAS=1.0 \
  LOG_DIR=logs/launch_centroid_continual \
  bash scripts/train/launch.sh gymnax_continual $ARMS

echo "=== $(date -Is) both blocks done ==="
echo "failures:"
grep -hv 'exit=0' logs/launch_centroid_*/status.tsv 2>/dev/null || echo "  none"
