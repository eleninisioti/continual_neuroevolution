#!/bin/bash
# ============================================================================
# MountainCar ONLY, at observation noise sigma=0.1, into the SAME tree as
# scripts/train/queue_iclr_centroid.sh: projects/iclr_2027/runs_centroid.
#
# WHY. The reported tree runs all three gymnax envs at one flat sigma=1.0, and
# on MountainCar that is not a sub-task, it is a wall. The observation is
# [position, velocity]: position spans 1.8 and velocity spans 0.14, so a
# N(0, 1.0^2) offset displaces velocity by ~7-15x its entire range. The task is
# about velocity. Every arm is being asked to control a dimension it can no
# longer read, and a comparison in which everything fails equally separates
# nothing.
#
# run_experiments.sh's own noise_range_for() already records this -- it defaults
# MountainCar to 0.02 against 2.0 for CartPole and Acrobot, citing Tang et al.
# (2025) A.1, who report 2.0 "is too large for this environment to be
# learnable". block_gymnax_continual bypasses that table and loops SIGMAS
# across all envs, which is how the flat 1.0 tree came to exist.
#
# 0.1 IS A MIDDLE, NOT A SOLVED PROBLEM. At 0.1 the velocity offset is ~0.03-0.08
# against a 0.14 span -- the same order as the signal rather than ten times it,
# which is what makes a sub-task hard-but-learnable instead of unobservable.
# It is still 5x Tang et al.'s 0.02. If 0.1 also floors every arm, the knob is
# right here: SIGMAS="0.02" (or "0.1 0.02") runs the next value into its own
# cell without touching anything already on disk.
#
# WHICH ARMS. Everything EXCEPT the two Iso+LineDD arms, `ga_isoline` and
# `dns`. That is not a shortcut, it is the one place the operator is known to
# be broken: on MountainCar at sigma=1.0, ga_isoline reaches float32's maximum
# on 10/10 trials and dns reaches max|w| ~1e13 (see
# queue_iclr_centroid_dns_gaussian.sh). Iso+LineDD multiplies population
# variance by (1 + 2*line_sigma^2) per generation and dominated novelty in an
# unbounded descriptor space rewards the outliers that produces; a saturated
# genotype is not a search, and re-running one at a gentler noise level would
# only produce a second broken cell. `dns_gaussian` carries dominated novelty
# into this cell at an operator that does not diverge. Override with
# ARMS="ga_isoline dns" to add them anyway.
#
# NO NONCONTINUAL BLOCK. Sub-task 0 is the unperturbed environment under every
# sigma, so the stationary phase does not depend on one -- and the tree records
# that: continual cells are tagged `MountainCar_v0_sigma<S>` while noncontinual
# cells are plain `MountainCar_v0`. The FT reference these runs need is already
# on disk for all eight arms. Training a second one would not be a control, it
# would be a duplicate at a different seed stream.
#
# THE CELL IS SEPARATE, SO NOTHING REPORTED MOVES. ENV_DIR_SUFFIX files these
# under `MountainCar_v0_sigma0.1`, and every analysis script selects cells by
# `--sigma` (make_lineplot.py, make_plasticity_figure.py,
# plasticity_checkpoints.py) or by an explicit `--envs` list
# (behavioural_divergence.py). The sigma=1.0 figures are untouched; see
# scripts/analysis/finish_iclr_centroid_mcar_sigma0.1.sh for the passes that
# read this one.
#
# THE SUB-TASK SEQUENCE IS SHARED, VERIFIED. GA and DNS inline their own noise
# draw while ES/NES and the RL arms call
# source/envs/gymnax_classic.task_noise_vectors; the two are bit-identical
# (same jax.random.key(trial*7919), same split order, no split for sub-task 0),
# so all eight arms face the same offsets at a given trial. CLAUDE.md rule (c).
#
# TWO IDLE CARDS BY DEFAULT. The trac and dns_gaussian launchers from the
# sigma=1.0 queue are still running and hold GPUs 0-3, 6 and 7; launch.sh
# leases GPUs per launcher, so a concurrent launcher must be given a DISJOINT
# set or the two will oversubscribe the same cards. GPUS=... overrides.
#
# SAFE TO RE-RUN: run_condition skips a (method, env, trial) whose
# training_metrics.json already exists, so an interrupted queue resumes.
#
#   bash scripts/train/queue_iclr_centroid_mcar_sigma0.1.sh
#
# Env: ARMS, GPUS, NUM_TRIALS (default 10), SIGMAS (default 0.1).
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ ! -f "$REPO_ROOT/pyproject.toml" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
cd "$REPO_ROOT"

ARMS="${ARMS:-ga dns_gaussian es nes ppo trac redo cchain}"
ROOT=projects/iclr_2027/runs_centroid
TRIALS="${NUM_TRIALS:-10}"
SIGMA="${SIGMAS:-0.1}"

# Four jobs per card, as in queue_iclr_centroid.sh: launch.sh leases one token
# per entry in GPUS, so repeating an index is how it is told to oversubscribe.
# A gymnax job is ~600 MiB, so eight fit on two cards with room to spare.
export GPUS="${GPUS:-4 4 4 4 5 5 5 5}"

echo "=== $(date -Is) MountainCar sigma $SIGMA continual block ==="
echo "    arms  : $ARMS"
echo "    gpus  : $GPUS"
echo "    cell  : $ROOT/gymnax/continual/<arm>/MountainCar_v0_sigma$SIGMA"

# Every setting other than ENVS and SIGMAS is copied from
# queue_iclr_centroid.sh, so this cell is comparable to the sigma=1.0 one:
# 10 trials from BASE_SEED 42, 20 sub-tasks at period 10, 200 generations each.
NUM_TRIALS="$TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  ENVS="MountainCar-v0" \
  NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 SIGMAS="$SIGMA" \
  LOG_DIR="logs/launch_centroid_mcar_sigma$SIGMA" \
  bash scripts/train/launch.sh gymnax_continual $ARMS

echo "=== $(date -Is) done ==="
echo "failures:"
grep -hv 'exit=0' "logs/launch_centroid_mcar_sigma$SIGMA/status.tsv" 2>/dev/null || echo "  none"
