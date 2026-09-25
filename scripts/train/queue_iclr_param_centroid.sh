#!/bin/bash
# ============================================================================
# The PHYSICS gymnax cell, RE-CUT FOR THE CENTROID, on a second machine.
#
# Into projects/iclr_2027/runs_param_centroid. This is the param counterpart of
# queue_iclr_centroid_mcar_sigma0.1_remote.sh and stands to
# projects/iclr_2027/runs_param exactly as runs_centroid stands to runs/:
# same experiment, re-run because the old tree cannot answer the question the
# figures now ask.
#
# WHY runs_param IS BEING THROWN AWAY AND NOT EXTENDED. Every NE trial in it
# was started before 2026-09-09 10:09, when the trainers began writing the
# centroid. Checked, not assumed:
#
#     runs_param/.../checkpoints.npz  ->  finalgen, incumbent, noise_vectors
#     runs_centroid/.../checkpoints.npz -> CENTROID, finalgen, incumbent, ...
#
# There is no `centroid` array and no `ne_centroid_*` metric anywhere under
# runs_param, so `make_lineplot.py --metric centroid` and the centroid
# plasticity figure would silently describe the ELITE there -- the same defect
# that retired most of the noise tree. `run_condition` skips any (method, env,
# trial) whose training_metrics.json already exists, so pointing this sweep at
# the old root would keep every stale NE trial and quietly produce a tree that
# verifies clean and is wrong. A new root is the only safe move.
#
# THE ARM LIST IS THE EIGHT REPORTED ARMS, not the nine runs_param queued.
# The Iso+LineDD column (`ga_isoline`, `dns`) is the operator ablation and the
# paper reports the gaussian column only, so it is dropped here as it was
# dropped from runs_centroid:
#
#     in    ga  dns_gaussian  es  nes  ppo  trac  redo  cchain
#     out   ga_isoline  dns
#
# `ga` is gaussian+truncation and `dns_gaussian` is gaussian+novelty, so the
# NE half of the figure is still a matched pair.
#
# NO NONCONTINUAL BLOCK. Sub-task 0 of a param run is the STOCK body, so the
# stationary phase of this experiment IS the stationary phase of the noise one
# -- same trainer, same seeds, same environment. It is on disk under
# runs_centroid/gymnax/noncontinual for all eight arms at 10 trials and is
# symlinked below rather than retrained (240 jobs saved). If runs_centroid is
# not on this machine the symlink is skipped with a warning; training does not
# need it, only the analysis does.
#
# `_sigma1.0` IN THE CELL NAMES IS A NAME, NOT A SETTING. No observation
# offset is drawn under `--task_type param` and `noise_range` is unused: the
# trainers carry an all-zero offset sequence so the saved artifacts have the
# same shape under both families. The suffix is kept because every analysis
# script recovers the environment by splitting the cell name on `_sigma`.
# What a run actually is, is in its own config as `task_type: param`.
#
# WHAT A SUB-TASK IS HERE. A log-uniform multiplier on one named group of the
# body's physics, observation untouched (source/utils/task_sequence.py,
# GYMNAX_PHYSICS_TASKS):
#
#     CartPole-v1      length (pole)       0.5x  - 2x
#     Acrobot-v1       mass (both links)   1/1.15x - 1.15x
#     MountainCar-v0   gravity             1/1.5x  - 1.5x
#
# Sub-task 0 is 1.0x and the draw is seeded from the TRIAL INDEX alone, so
# every method at a given trial meets the same bodies in the same order
# (CLAUDE.md rule (c)).
#
# EVERY BUDGET SETTING IS SPELLED OUT BELOW AND NOT INHERITED. A queue that
# sets only PROJECT_ROOT picks up block_gymnax_continual's defaults
# (NUM_TASKS=10, TASK_PERIOD=0) and writes half-budget runs over sub-tasks
# that are never revisited -- so no forgetting is measurable at all -- into a
# cell whose other arms are at the full budget. That has now happened three
# times in this project and every batch had to be thrown away.
#
#   3.072e9 env steps per trial   = 4000 gens x 512 pop x 3 evals x 500 steps
#                                 = the RL arms' --num_timesteps
#   boundary every 1.536e8 steps  = 200 gens = 1500 PPO updates
#   20 sub-tasks at period 10     = 10 distinct bodies, each visited twice
#
# ---------------------------------------------------------------------------
# USAGE, on the second machine, from the repo root
#
#     bash scripts/train/queue_iclr_param_centroid.sh
#
# Safe to re-run and safe to split across machines: run_condition skips any
# (method, env, trial) already on disk, so two machines with disjoint ARMS
# write disjoint directories. Cost is ~80 jobs; on the sigma=1.0 noise cell the
# measured medians were nes 154 min, cchain 137, trac 132, redo 101, ppo 71,
# ga 65, es 62, dns_gaussian ~60, so the RL arms dominate the wall clock. Split
# them off with ARMS= if a third machine is free.
#
# Bring the results home with:
#
#     rsync -av --ignore-existing \
#         projects/iclr_2027/runs_param_centroid/gymnax/continual/ \
#         <home>:<repo>/projects/iclr_2027/runs_param_centroid/gymnax/continual/
#
# Then, at home: bash scripts/analysis/finish_iclr_param.sh
#
# Env: ARMS, ENVS, GPUS, NUM_TRIALS (default 10), JOBS_PER_GPU (default 4).
#
# DO NOT EDIT run_experiments.sh OR launch.sh IN PLACE WHILE THIS RUNS. bash
# reads a script incrementally, so an in-place rewrite under a live launcher
# corrupts its parse. Write a temp file and `mv` it.
# WHAT THIS CELL IS FOR, DECIDED 2026-09-09: THE APPENDIX. It is a negative
# result and should be queued knowing that, not discovered again downstream.
# Measured on the old `runs_param` tree and on a widened pilot:
#
#   * THE PHYSICS FAMILY IS NEAR-SATURATED. Zero-shot on an UNSEEN body, mean
#     over sub-tasks 1-9: CartPole ga 427 against an end-of-sub-task 499, es
#     477/500, dns 412/467; Acrobot ga -88/-80; MountainCar ga -135/-119. The
#     observation-noise tree at the same budget puts CartPole zero-shot at
#     26-79 against 269-443. A new body costs a few percent; a new offset costs
#     most of the score.
#   * ZERO-SHOT ON A REVISITED SUB-TASK IS NO BETTER THAN ON A FRESH ONE, so
#     there is no retention signal here to separate methods with. That is the
#     whole reason this is an appendix.
#   * RL DOES NOT STRUGGLE HERE EITHER. Where PPO has data it sits among the NE
#     arms (CartPole zero-shot 438 against ga 428 / es 477). Do not quote an
#     NE-vs-RL claim off this cell.
#   * WIDENING THE RANGE DOES NOT FIX IT. At 0.25x-4x on CartPole the zero-shot
#     dip becomes real (170 at 2.66x, 191 at 0.30x) but PPO is back at 500
#     within TEN updates, and by sub-task 3 it zero-shots the next body at
#     exactly 500 -- it has become a generalist over pole length. The axis is
#     COMPATIBLE: one policy covers the whole range, so widening stretches the
#     transient without creating interference. Acrobot cannot be widened at all
#     (a specialist at 2x mass tops out near -138) and MountainCar's rescalings
#     are nested. That pilot is `queue_param_wide_ppo_pilot.sh`.
#
# The family that DOES bite is `--task_type actions`, where the two regimes
# demand opposite outputs at the same input and no generalist exists:
# `queue_actions_full.sh`. This cell is its control -- the same protocol on an
# axis with a generalist -- which is exactly what an appendix is for.
#
# PPO's dormancy is worth keeping from this cell even so: it climbs 13->48 %
# policy and 41->75 % value here, the SAME trajectory as the noise cell where
# it collapses to 247, while performance stays pinned at 500. Same dormancy,
# opposite outcome -- the sharpest evidence in the project that the dormant
# fraction tracks convergence, not failure.
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ ! -f "$REPO_ROOT/pyproject.toml" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
cd "$REPO_ROOT"

ARMS="${ARMS:-ga dns_gaussian es nes ppo trac redo cchain}"
ENVS="${ENVS:-CartPole-v1 Acrobot-v1 MountainCar-v0}"
ROOT=projects/iclr_2027/runs_param_centroid
TRIALS="${NUM_TRIALS:-10}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"

# ------------------------------------------------------------------ preflight
# The ENTIRE POINT of this sweep is the centroid, so refuse to start on a
# checkout that predates it rather than spend 80 GPU-hours reproducing the
# defect. All three NE trainers must pass `centroid=` to save_eval_artifacts;
# under `param` that call also has to be reached, which it was not before
# 2026-09-08 (it was gated on task_type == 'noise').
missing=""
for f in train_GA_gymnax_continual train_ES_gymnax_continual train_DNS_gymnax_continual; do
    grep -q 'centroid=ckpt_centroid' "source/studies/gymnax/$f.py" || missing="$missing $f"
    grep -q "param_mults=task_param_values" "source/studies/gymnax/$f.py" || missing="$missing $f:param"
done
if [ -n "$missing" ]; then
    echo "REFUSING TO LAUNCH: this checkout does not write the centroid under" >&2
    echo "  --task_type param. Missing in:$missing" >&2
    echo "  Pull the 2026-09-09 trainers before queueing." >&2
    exit 1
fi

mkdir -p "$ROOT/gymnax"

# ------------------------------------------------------- noncontinual block
# Linked, not retrained -- see the header. A relative symlink so the tree
# survives being rsynced to another path.
NONCONT_TARGET="$REPO_ROOT/projects/iclr_2027/runs_centroid/gymnax/noncontinual"
if [ ! -e "$ROOT/gymnax/noncontinual" ]; then
    if [ -d "$NONCONT_TARGET" ]; then
        ln -s ../../runs_centroid/gymnax/noncontinual "$ROOT/gymnax/noncontinual"
        echo "linked noncontinual -> runs_centroid/gymnax/noncontinual"
    else
        echo "NOTE: runs_centroid/gymnax/noncontinual is not on this machine."
        echo "      Training does not need it; make the symlink at home before"
        echo "      running finish_iclr_param.sh, which reads it for the FT column."
    fi
fi

# ---------------------------------------------------------- GPU oversubscribe
# launch.sh leases one token per entry in GPUS, so REPEATING an index is how it
# is told to put more than one job on a card. A gymnax job is ~600 MiB; four fit
# on anything with 8 GB. These are new runs where only the statistics matter, so
# sharing a card is fine -- MJX and GPU PPO are not bit-reproducible anyway.
if [ -z "${GPUS:-}" ]; then
    VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')
    VISIBLE=${VISIBLE:-0}
    GPUS=""
    for g in $VISIBLE; do
        for _ in $(seq 1 "$JOBS_PER_GPU"); do GPUS="$GPUS $g"; done
    done
    GPUS="${GPUS# }"
fi
export GPUS

echo "=== $(date -Is) gymnax PHYSICS continual block (centroid re-cut) ==="
echo "    root  : $ROOT"
echo "    arms  : $ARMS"
echo "    envs  : $ENVS"
echo "    gpus  : $GPUS"
echo "    trials: 1..$TRIALS"
echo "    task  : param, 20 sub-tasks, period 10, 200 gens each"

NUM_TRIALS="$TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  ENVS="$ENVS" \
  TASK_TYPE=param NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 SIGMAS=1.0 \
  LOG_DIR=logs/launch_param_centroid \
  bash scripts/train/launch.sh gymnax_continual $ARMS

echo "=== $(date -Is) done ==="
echo "failures:"
grep -hv 'exit=0' logs/launch_param_centroid/status.tsv 2>/dev/null || echo "  none"
