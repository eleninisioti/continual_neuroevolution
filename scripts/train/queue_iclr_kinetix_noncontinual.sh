#!/bin/bash
# ============================================================================
# THE KINETIX STATIONARY BLOCK. Run this FIRST, on the other server.
#
# Twenty hand-designed Kinetix levels, one stationary run each, per arm. It
# answers the question that has to be answered before any continual claim:
# CAN EACH METHOD SOLVE A LEVEL AT ALL, cold, at this budget? A retention
# number for a method that never learned the level says nothing.
#
# Levels nobody solves are not a failure of this block, they are its output.
# The continual chain warm-starts each level from the one before it, so a level
# out of reach cold may be in reach warm -- and knowing WHICH levels those are
# is the interesting half.
#
# WHAT IT REPRODUCES
#
# `projects/kinetix/budget_g200_r3_gafinal` is the run that mattered: a GA
# that solved all twenty levels in the continual chain, on the PREVIOUS
# codebase. Every environment- and search-side value of that run is reproduced
# here -- population 512, 200 generations a level, one rollout a genome of 128
# steps at frame skip 2 (that run's 3 x 256 were measured inert, see
# settings.py, 2026-09-13), mutation sigma 0.5, crossover 0.2, an archive of
# 256, and the 1,128,256-parameter actor-only pixel network, which
# `source/envs/kinetix.py:build_policy` ASSERTS rather than reports. `es` and
# `dns` likewise take `budget_g200_r3_esfull_s1`'s and
# `budget_g200_r3_dnsfix_s1`'s own settings.
#
# What is NOT reproduced, deliberately, is the old table's compute gap: its RL
# arms got 3.2e6 environment steps a level against NE's 7.9e7. Here both
# families get 1.311e7 -- 200 x 512 x 1 x 128 for NE, 1600 x 128 x 64 for RL --
# and `source/studies/kinetix/settings.py:check()` recomputes both at the start
# of every trial and refuses to run if they have drifted (CLAUDE.md (c)).
#
# THE GRID
#
#     cells   Kinetix-h0_unicycle .. Kinetix-h19_thrust_left_very_easy  (20)
#     arms    ga es nes dns dns_gaussian ppo trac redo cchain
#     trials  1..NUM_TRIALS  (default 1 -- this is a shakeout, not the table)
#
# `ga` is THE arm this block exists to check. `ga` and `dns_gaussian` are the
# GAUSSIAN pair -- identical mutation, so they differ in the selection rule
# alone. `dns` is the old kinetix DNS (Iso+LineDD), kept because it is the arm
# the previous table reported.
#
# `ppo` is here on evidence: from scratch on h0_unicycle it reaches the solved
# return by update 100 -- 819k environment steps, against the 1.6e6 the
# previous codebase's four from-scratch runs took.
#
# EVERY ARM OF THE CONTINUAL BLOCK IS HERE, TRAC / ReDo / C-CHAIN INCLUDED, and
# running a continual method where nothing changes is the point of it, twice
# over:
#
#   1. FT. Forward transfer subtracts each method's OWN stationary curve from
#      its continual one. Take C-CHAIN's stationary runs away and C-CHAIN has
#      no FT column -- and substituting PPO's would charge the mechanism for
#      whatever the mechanism costs on a fixed level, which is exactly the
#      quantity FT is supposed to isolate.
#   2. The plasticity control. A dormancy, churn or rank number is evidence
#      about task CHANGES only if the same measurement, on a run of the same
#      length with no change in it, does something different. That control has
#      to be the same arm: ReDo's dormancy against PPO's dormancy is a
#      comparison between methods, not between a switching run and a stationary
#      one.
#
# So the arm sets of the two blocks match, and they should stay matched.
#
# NO ARM IS TOLD WHERE A PHASE BOUNDARY IS (CLAUDE.md (d)) -- and on this block
# there is nothing to tell: the level never changes. The ten-phase checkpoint
# grid still exists, so a dormancy or churn number here is the no-task-change
# control for the same measurement on the continual chain.
#
# USAGE
#   git pull                                    # the guards below need it
#   DRY_RUN=1 bash scripts/train/queue_iclr_kinetix_noncontinual.sh
#   nohup bash scripts/train/queue_iclr_kinetix_noncontinual.sh \
#       > logs/kinetix_noncontinual.log 2>&1 &
#
#   # a single level first, if you want one number before committing the card:
#   CELLS=Kinetix-h0_unicycle ARMS=ga \
#       bash scripts/train/queue_iclr_kinetix_noncontinual.sh
#
#   # when it is done, send the tree home
#   rsync -av --ignore-existing \
#       projects/iclr_2027/runs_kinetix/kinetix/ \
#       <home>:<repo>/projects/iclr_2027/runs_kinetix/kinetix/
#
# COST. On the 2026-08 runs a GA level took 63 min and an OpenES level 77 min,
# one job to a card, at this exact population and episode length; the RL arms
# are budget-matched to them, so assume the same order (C-CHAIN is the
# exception at ~2.7x PPO's per-update cost). 9 arms x 20 levels x 1 trial is
# 180 jobs, so roughly 210 job-hours: about 26 h on 8 cards at one job each.
#
# If that is too long for a first pass, cut it by CELLS or by arm, not by
# dropping the continual-RL arms -- they are the FT reference for three rows of
# the continual table. A sensible split is the five NE arms plus ppo first,
# then `ARMS="trac redo cchain"` behind it, which is the same 180 jobs in two
# waves and gets the GA answer sooner. JOBS_PER_GPU is 1 because a job carries 512 x 1.13M
# parameters (2.3 GB for the population alone) plus a chunked pixel rollout;
# raise it once `nvidia-smi` has shown you what one job actually costs on your
# cards. NUM_TRIALS=1 is the shakeout; the reported table needs 3+.
#
# Env: GPUS, JOBS_PER_GPU (1), NUM_TRIALS (1), ROOT, ARMS, CELLS, DRY_RUN.
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
PYTHON="${PYTHON:-.venv/bin/python}"
LOGS=logs/kinetix_noncontinual
mkdir -p "$LOGS"

# ------------------------------------------------------------ version guard
# The same check whose absence cost eight GA trials on 2026-09-09: a launcher
# started against a checkout whose trainer could not yet write what the figures
# read, and Python reads its source at process START, so runs that FINISH after
# a fix can still lack it.
if [ ! -f source/studies/kinetix/cli.py ] || [ ! -f source/envs/kinetix.py ]; then
    echo "FATAL: this checkout has no Kinetix study. Run: git pull" >&2
    exit 1
fi
if ! grep -q 'kinetix_noncontinual)' scripts/train/run_experiments.sh; then
    echo "FATAL: run_experiments.sh has no kinetix blocks. Run: git pull" >&2
    exit 1
fi
for f in train_nes train_ppo; do
    if ! grep -q 'save_checkpoints' "source/studies/generalists/${f}.py"; then
        echo "FATAL: source/studies/generalists/${f}.py cannot write" >&2
        echo "       checkpoints.npz. This checkout is too old. Run: git pull" >&2
        exit 1
    fi
done

# The level files have to be on disk -- the vendored Kinetix checkout ships
# them, and a fresh clone without its submodule/LFS payload fails here rather
# than 100 times inside the launcher.
if ! $PYTHON -c "
from kinetix.util.saving import load_from_json_file
from source.envs.kinetix_levels import LEVELS
for lvl in LEVELS:
    load_from_json_file('m/' + lvl)
print('all 20 level files load')
"; then
    echo "FATAL: the Kinetix level files are missing or kinetix is not" >&2
    echo "       importable. Check third_party/kinetix and the install." >&2
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

# A real import of the settings, which is also the compute-match assertion.
if ! $PYTHON source/studies/kinetix/cli.py \
        --env Kinetix-h0_unicycle --method ga \
        --output_dir /tmp/kinetix_preflight --dry_run; then
    echo "FATAL: preflight failed -- settings or environment are broken." >&2
    exit 1
fi
echo "guards OK"

CELLS="${CELLS:-$($PYTHON -c "
from source.envs.kinetix_levels import LEVELS, cell_for
print(' '.join(cell_for(l) for l in LEVELS))")}"

echo "root       : $ROOT/kinetix/noncontinual"
echo "cells      : $(echo $CELLS | wc -w) levels"
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
      PROJECT_ROOT="$ROOT" LOG_DIR="$LOGS" PYTHON="$PYTHON" \
      bash scripts/train/launch.sh kinetix_noncontinual $ARMS
    echo "=== DRY RUN -- nothing queued ==="
    exit 0
fi

echo "=== $(date -Is) starting ==="
GPUS="$GPUS" NUM_TRIALS="$NUM_TRIALS" ENVS="$CELLS" \
  PROJECT_ROOT="$ROOT" LOG_DIR="$LOGS" PYTHON="$PYTHON" \
  bash scripts/train/launch.sh kinetix_noncontinual $ARMS
code=$?
echo "=== $(date -Is) done (exit=$code) ==="
echo "failures:"
grep -v 'exit=0' "$LOGS/status.tsv" 2>/dev/null || echo "  none"
