#!/bin/bash
# =============================================================================
# `dns_gaussian` into projects/iclr_2027/runs_centroid, both phases.
#
# WHY THIS ARM EXISTS
# -------------------
# Iso+LineDD displaces a child along the line between its two parents, so its
# step scales with the population's own spread and compounds -- the operator
# multiplies population variance by (1 + 2*line_sigma^2) per generation, and
# dominated novelty in an unbounded descriptor space rewards the outliers that
# growth produces rather than contracting them. Measured on the sigma=1.0
# continual tree, max|w| per trial:
#
#     gaussian arms   ga 5.6e1-9.2e1,  es ~1e1,  nes ~1e0,  ppo ~1e1
#     Iso+LineDD      dns 9.6e10-1.5e13,  ga_isoline 7.4e12-3.4e38
#
# `ga_isoline` reaches float32's maximum on 10/10 MountainCar trials and 1/10
# on Acrobot: those genotypes stopped at the representable limit, so their
# weight row is a ceiling and their step norm is 0 because consecutive
# checkpoints are the same saturated vector.
#
# The divergence buys nothing. Across DNS trials, max|w| correlates with
# neither performance nor diversity on any task (|rho| <= 0.49, p >= 0.15,
# n = 10 per task), and the sign is not even consistent between tasks.
#
# WHAT IT COMPLETES
# -----------------
# The 2x2 of {gaussian, Iso+LineDD} x {fitness truncation, dominated novelty}:
#
#                     gaussian          Iso+LineDD
#     truncation      ga                ga_isoline
#     novelty         dns_gaussian      dns          <- this script adds the gap
#
# Without it, "DNS beats the GA" confounds the selection rule with the
# recombination operator, and the operator-matched control (`ga_isoline`) is
# the arm that overflows -- so on MountainCar that comparison is currently
# between two saturated searches and says nothing about novelty selection.
#
# NOT A CLIP. qdax's isoline_variation takes optional minval/maxval ("Back in
# bounds if necessary (floating point issues)") and we never pass them, but a
# clipped genotype sitting at the bound is still not a search. Changing the
# operator is the test; bounding it would only hide the compounding.
#
# BOTH PHASES. FT subtracts a method's OWN stationary run, and `dns_gaussian`
# is a different search from `dns`, so it needs its own noncontinual reference
# exactly as `ga_isoline` does.
#
#   bash scripts/train/queue_iclr_centroid_dns_gaussian.sh
#
# Env: GPUS (default all), NUM_TRIALS (default 10), ENVS (continual phase only,
# default "CartPole-v1 Acrobot-v1" -- see the note at that block).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ ! -f "$REPO_ROOT/pyproject.toml" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
cd "$REPO_ROOT"

export PROJECT_ROOT="projects/iclr_2027/runs_centroid"
export NUM_TRIALS="${NUM_TRIALS:-10}"
export SIGMAS="1.0"

# THE SUB-TASK STRUCTURE, WHICH THIS SCRIPT USED TO OMIT AND WHICH IS NOT
# OPTIONAL. block_gymnax_continual DEFAULTS to NUM_TASKS=10 and TASK_PERIOD=0,
# so a queue that sets only PROJECT_ROOT/NUM_TRIALS/SIGMAS trains 2000
# generations of ten sub-tasks that are never revisited -- half the budget of
# every other arm in the cell, and a sequence in which forgetting cannot be
# measured at all. That is exactly what happened here: all 10 CartPole and all
# 10 Acrobot dns_gaussian trials landed at (1.536e9 steps, period 0) inside a
# cell whose other nine arms are at (3.072e9, period 10), and they had to be
# deleted and retrained. A retry or a single-arm queue must repeat EVERY
# setting of the queue it is completing.
export NUM_TASKS=20
export TASK_PERIOD=10
export TASK_INTERVAL=200
# On for every other arm in this cell; an arm without it is a hole in the
# diversity panel rather than a cheaper run.
export TRACK_DIVERSITY=1

echo "=== dns_gaussian -> $PROJECT_ROOT (${NUM_TRIALS} trials, sigma 1.0) ==="

# The stationary reference first: it is the cheaper phase and FT is undefined
# without it, so finishing it first means a partial continual phase is already
# reportable.
LOG_DIR=logs/launch_centroid_dnsg_noncontinual \
    bash scripts/train/launch.sh gymnax_noncontinual dns_gaussian

# CARTPOLE AND ACROBOT ONLY. MountainCar at sigma=1.0 is not a sub-task, it is
# a wall: the observation is [position, velocity], velocity spans 0.14, and a
# N(0, 1.0^2) offset displaces it by ~7-15x its entire range, so every arm is
# asked to control a dimension it cannot read. That is the whole reason
# queue_iclr_centroid_mcar_sigma0.1.sh exists, and dns_gaussian is one of the
# eight arms in that cell -- so dominated novelty on gaussian mutation IS
# covered on MountainCar, at the noise level where the task is still learnable.
# Running it here as well would only add a third saturated MountainCar cell
# next to `dns` and `ga_isoline`.
#
# The noncontinual block above keeps all three envs: sub-task 0 is the
# unperturbed environment under every sigma, so its MountainCar runs are the FT
# reference the sigma=0.1 cell needs. Do not narrow that one.
ENVS="${ENVS:-CartPole-v1 Acrobot-v1}" \
LOG_DIR=logs/launch_centroid_dnsg_continual \
    bash scripts/train/launch.sh gymnax_continual dns_gaussian

echo "=== done; now run the post-hoc passes, see scripts/README.md ==="
