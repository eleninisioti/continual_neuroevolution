#!/bin/bash
# ============================================================================
# THE KINETIX CONTINUAL BLOCK. Run this SECOND, after the stationary one.
#
# One cell, `Kinetix20`: the twenty hand-designed levels visited in order,
# 200 generations each, 4000 generations end to end. This is the chain the
# previous codebase's `budget_g200_r3_gafinal` ran, where the GA solved all
# twenty -- the result this whole port exists to reproduce on the shared
# runners.
#
# DO NOT RUN THIS BEFORE THE STATIONARY BLOCK HAS SAID SOMETHING. A retention
# or forward-transfer number needs each method's own stationary curve to
# subtract, and a chain in which an arm never learns any level is 21 hours of
# card time that answers nothing. `queue_iclr_kinetix_noncontinual.sh` is the
# prerequisite, and the guard below refuses to start until its tree exists.
#
# COMPUTE-MATCHED (CLAUDE.md (c)): 2.621e8 environment steps for both families
# -- 4000 generations x 512 x 1 x 128 for NE, 32,000 updates x 128 x 64 for
# RL (one rollout of 128 steps since 2026-09-13; settings.py says why the
# 3 x 256 nominal budget was six times what the search used) -- cut into the
# SAME twenty phases, so both meet a level change at the
# same point on the shared wall of steps. `settings.check()` re-derives it at
# the start of every trial.
#
# NO ARM IS TOLD WHERE A LEVEL CHANGES (CLAUDE.md (d)). The GA and DNS
# re-evaluate their stored population every generation rather than only at a
# transition, C-CHAIN's `cchain_reset_on_switch` is off, and nothing here
# passes a boundary signal. `KINETIX_EXTRA=--oracle` builds the deliberately
# informed control and is the only way to turn any of it on.
#
# USAGE
#   nohup bash scripts/train/queue_iclr_kinetix_continual.sh \
#       > logs/kinetix_continual.log 2>&1 &
#
# COST. The 2026-08 chain took 21 h a trial for the GA, one job to a card, at
# this exact configuration. 9 arms x 1 trial is 9 jobs, so one pass fits on 9
# cards in a day; NUM_TRIALS=3 is what a reported row needs and is three days
# on the same 9 cards. C-CHAIN is the slow one, ~2.7x PPO per update.
#
# The arm set MATCHES the stationary block's on purpose: every arm here needs
# its stationary twin to have an FT column and a no-task-change control. Do not
# add an arm to one block without adding it to the other.
#
# A KILLED KINETIX NE RUN LOSES THE CHAIN -- the artifacts are written at the
# end. Stop it with `touch $ROOT/STOP` or by letting it finish, not with a
# kill.
#
# Env: GPUS, JOBS_PER_GPU (1), NUM_TRIALS (1), ROOT, ARMS, DRY_RUN.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# The corrected-budget tree (2026-09-13). `runs_kinetix` is the 256-step,
# 3-rollout tree with RL at 9600 updates a level, kept as the nominal-budget
# reference and not reported.
ROOT="${ROOT:-projects/iclr_2027/runs_kinetix_ep128_ev1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
NUM_TRIALS="${NUM_TRIALS:-1}"
DRY_RUN="${DRY_RUN:-0}"
ARMS="${ARMS:-ga es nes dns dns_gaussian ppo trac redo cchain}"
CELLS="${CELLS:-Kinetix20}"
PYTHON="${PYTHON:-.venv/bin/python}"
LOGS=logs/kinetix_continual
mkdir -p "$LOGS"

if [ ! -f source/studies/kinetix/cli.py ] || [ ! -f source/envs/kinetix.py ]; then
    echo "FATAL: this checkout has no Kinetix study. Run: git pull" >&2
    exit 1
fi
if ! grep -q 'kinetix_continual)' scripts/train/run_experiments.sh; then
    echo "FATAL: run_experiments.sh has no kinetix blocks. Run: git pull" >&2
    exit 1
fi

# The prerequisite, checked rather than trusted: without the stationary tree
# there is nothing to read this block against.
if [ "${SKIP_STATIONARY_CHECK:-0}" = "0" ] \
   && [ ! -d "$ROOT/kinetix/noncontinual" ]; then
    echo "FATAL: $ROOT/kinetix/noncontinual does not exist." >&2
    echo "       Run queue_iclr_kinetix_noncontinual.sh first, or set" >&2
    echo "       SKIP_STATIONARY_CHECK=1 if you know why you are skipping it." >&2
    exit 1
fi

# The multi-discrete head's gradients. This is a REGRESSION GUARD, not a
# formality: the head shipped on 2026-09-09 padded its ragged logit block with
# -inf, which gave every value correctly and a NaN gradient, and PPO's entropy
# bonus then NaN'd the actor on the first step. It reported itself as
# `H=0.000` with the return at the floor -- an entropy collapse, not a NaN --
# and survived a bisect over batch shape, learning rate and Adam epsilon. It
# costs a second here and it cost most of a day there.
if ! $PYTHON -c "
from source.studies.generalists.actors import _multi_discrete_grads_finite
_multi_discrete_grads_finite()
print('multi-discrete head gradients finite')
"; then
    echo "FATAL: the multi-discrete action head produces non-finite" >&2
    echo "       gradients; PPO on this body would train on NaN." >&2
    exit 1
fi

if ! $PYTHON source/studies/kinetix/cli.py \
        --env Kinetix20 --method ga \
        --output_dir /tmp/kinetix_preflight --dry_run; then
    echo "FATAL: preflight failed -- settings or environment are broken." >&2
    exit 1
fi
echo "guards OK"

echo "root       : $ROOT/kinetix/continual"
echo "cells      : $CELLS"
echo "arms       : $ARMS"
echo "trials     : 1..$NUM_TRIALS"
echo "jobs       : $(( $(echo $ARMS | wc -w) * $(echo $CELLS | wc -w) * NUM_TRIALS ))"

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
      PROJECT_ROOT="$ROOT" LOG_DIR="$LOGS" PYTHON="$PYTHON" \
      bash scripts/train/launch.sh kinetix_continual $ARMS
    echo "=== DRY RUN -- nothing queued ==="
    exit 0
fi

echo "=== $(date -Is) starting ==="
GPUS="$GPUS" NUM_TRIALS="$NUM_TRIALS" ENVS="$CELLS" \
  PROJECT_ROOT="$ROOT" LOG_DIR="$LOGS" PYTHON="$PYTHON" \
  bash scripts/train/launch.sh kinetix_continual $ARMS
code=$?
echo "=== $(date -Is) done (exit=$code) ==="
echo "failures:"
grep -v 'exit=0' "$LOGS/status.tsv" 2>/dev/null || echo "  none"
