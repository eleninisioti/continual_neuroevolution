#!/bin/bash
# ============================================================================
# THE ACTION-REVERSAL SWEEP: all eight reported arms, all three gymnax envs,
# on a second machine. Into projects/iclr_2027/runs_actions.
#
# WHY THIS FAMILY AND NOT THE OTHER TWO. Both existing families are COMPATIBLE
# -- one policy covers every sub-task, so a switch is absorbed rather than
# relearned, and there is nothing for a continual-learning method to be good at:
#
#   physics  PPO pinned at 500 through all 20 sub-tasks; zero-shot on an unseen
#            body 438 against an end-of-sub-task 493. WIDENED to 0.25x-4.0x the
#            zero-shot dip becomes real (170 at 2.66x, 191 at 0.30x) but PPO is
#            back at 500 within TEN updates, and by sub-task 3 it zero-shots the
#            next body at exactly 500 -- it has become a generalist over pole
#            length. Widening stretched the transient, it created no
#            interference. Acrobot cannot be widened at all (a specialist at 2x
#            mass tops out near -138) and MountainCar's rescalings are nested.
#   noise    conflicting enough that PPO collapses, but the conflict is in the
#            OBSERVATION, so it is confounded with "the input distribution
#            moved".
#
# WHAT THE 2026-09-09 PILOT ALREADY MEASURED, so this sweep is confirmation at
# 10 trials and not exploration. Read it before reading the figures:
#
#   * NOTHING RETAINS ANYTHING, and the two families fail oppositely. GA, DNS and
#     PPO solve BOTH regimes and retain 0-1 %. ES and NES solve ONLY the stock
#     regime -- the floor on every reversed sub-task -- and therefore "retain"
#     98-99 %, which is the never-learned-it artifact and not a result. Mean over
#     all sub-tasks on Acrobot: GA -96, DNS -87, ES/NES -290.
#   * END-OF-SUB-TASK SATURATES for every arm that crosses at all, so the
#     lineplot metric separates nothing here. The reportable columns are
#     RETENTION (zero-shot on a revisit) and the diversity/displacement
#     diagnostics below.
#   * ES/NES halt because their population fitness variance is EXACTLY 0.000 on
#     reversed Acrobot -- at sigma 0.2, 0.5 and 1.0 alike. Not a local optimum: a
#     local optimum still has a gradient pointing back. Sigma cannot fix it.
#   * GA and DNS cross because of POPULATION WIDTH. Genomic diversity at the
#     switch: nes 2.8, es 5.9, ga 54-61, dns 45-48; fitness std 0.00/0.72 against
#     20.0/3.3 and 152/80. Monotone, and GA reaches that width from truncation
#     selection plus gaussian mutation alone, so novelty search is not required.
#   * THE C.2 PLASTICITY ORDERING INVERTS. Displacement slowdown on Acrobot:
#     NES 3.95, ES 1.71, PPO 1.61, DNS 0.92 -- NES halts, PPO does not. The
#     generalists report's noise-family numbers are PPO 49 and 18, NES 0.26.
#     Same metric, opposite ordering. This is the headline the sweep exists to
#     confirm at n=10.
#
# TRACK_DIVERSITY=1 and BEHAVIOUR_SNAPSHOTS are therefore NOT optional here: the
# bd_genomic_diversity and bd_fitness_std columns ARE the result, not a
# diagnostic beside it. BEHAVIOUR_SNAPSHOTS counts snapshots per sub-task, so
# the default 8 already samples each switch.
#
# Under `actions` sub-task i reverses the action order (a -> n-1-a) and leaves
# the observation and the body untouched. For a memoryless policy of the
# observation -- the NE MLP and PPO's actor alike -- the two regimes demand
# OPPOSITE outputs at the same input, so there is provably no single policy
# that scores on both. A method cannot interpolate across this boundary; it can
# only relearn. Measured on the stock body: a policy at 500 under the stock
# order scores 9.0 zero-shot under the reversed one, and after training on
# reversed scores 8.9 back on stock. Both directions total.
#
# Reversal is well defined on all three bodies: CartPole left<->right, Acrobot
# torque -1<->+1 (0 fixed), MountainCar push-left<->push-right.
#
# MOUNTAINCAR IS THE WEAK CELL and may not be reportable. In the pilot every NE
# arm ended below its -150 threshold on BOTH regimes (GA -222/-179, ES -267/-500)
# so its rows measure undertraining rather than retention. It is kept because
# this sweep runs at ten times the pilot's per-sub-task budget and that may be
# enough; check the end-of-sub-task column before quoting anything from it.
#
# THE SEQUENCE ALTERNATES AND IS NOT DRAWN FROM THE TRIAL SEED. A reversal flag
# has two states, so sampling it would give runs of consecutive sub-tasks in the
# same regime -- boundaries at which nothing changes -- and a different number
# of real switches per trial. Alternation makes every one of the 19 boundaries a
# real reversal and gives every trial the same number, which is what CLAUDE.md
# rule (c) asks for. Sub-task 0 is the stock order, so it is the noncontinual
# experiment exactly as under the other two families. Every method at a given
# trial therefore meets the same regimes at the same steps; only the training
# seed differs. See `action_flip_sequence` in source/utils/task_sequence.py.
#
# NO CUE, DELIBERATELY. `TaskSpec`'s `actions_cue` adds the trial's observation
# offset to the reversed sub-task, which makes the regime identifiable from the
# observation and a generalist possible again. That is a different and strictly
# easier experiment; run it as the contrast if this one turns out so hard that
# nothing separates.
#
# THIS NEEDED CODE, AND THE CODE IS IN THE COMMIT YOU ARE PULLING. `FlipEnv`
# and `TaskSpec` existed in source/envs/gymnax_classic.py but only the
# generalists and brax paths used them; the gymnax trainers imported
# `apply_physics` alone and had no `actions` branch. Added 2026-09-09 to all
# four trainers, to save_eval_artifacts (`action_flips`, beside `param_mults`)
# and to evaluate_continual.py, which now rebuilds a reversed sub-task the way
# it rebuilds a rescaled body. A checkout without those commits will exit 2 on
# `--task_type actions`; the preflight below refuses to launch on one.
#
# NO NONCONTINUAL BLOCK. Sub-task 0 is the stock action order on the stock
# body, so the stationary phase of this experiment IS the stationary phase of
# the noise one -- same trainer, same seeds, same environment. It is on disk
# under runs_centroid/gymnax/noncontinual for all eight arms at 10 trials and
# is symlinked below rather than retrained (240 jobs saved). If runs_centroid is
# not on this machine the symlink is skipped with a warning; training does not
# need it, only the analysis does.
#
# `_sigma1.0` IN THE CELL NAMES IS A NAME, NOT A SETTING, for the third time in
# this project: no observation offset is drawn under `--task_type actions` and
# `noise_range` is unused. The suffix is kept because every analysis script
# recovers the environment by splitting the cell name on `_sigma`. What a run
# actually is, is in its own config as `task_type: actions`.
#
# THE ARM LIST IS THE EIGHT REPORTED ARMS. The Iso+LineDD column (`ga_isoline`,
# `dns`) is the operator ablation and the paper reports the gaussian column
# only, so `ga` (gaussian+truncation) and `dns_gaussian` (gaussian+novelty) are
# the NE pair here, as in runs_centroid.
#
# EVERY BUDGET SETTING IS SPELLED OUT AND NOT INHERITED. A queue that sets only
# PROJECT_ROOT picks up block_gymnax_continual's defaults (NUM_TASKS=10,
# TASK_PERIOD=0) and writes half-budget runs over sub-tasks that are never
# revisited, into a cell whose other arms are at the full budget. That has
# happened three times in this project and every batch was thrown away.
#
#   3.072e9 env steps per trial   = 4000 gens x 512 pop x 3 evals x 500 steps
#                                 = the RL arms' --num_timesteps
#   boundary every 1.536e8 steps  = 200 gens = 1500 PPO updates
#   20 sub-tasks at period 10     = the two regimes, ten visits each
#
# ---------------------------------------------------------------------------
# USAGE, on the second machine, from the repo root, after pulling
#
#     bash scripts/train/queue_actions_full.sh
#
# 240 jobs. On the sigma=1.0 noise cell the measured per-trial medians were
# nes 154 min, cchain 137, trac 132, redo 101, ppo 71, ga 65, es 62,
# dns_gaussian ~60, so the RL arms dominate the wall clock; splitting them onto
# a third machine with ARMS= roughly halves it. Safe to re-run and safe to
# overlap: run_condition skips any (method, env, trial) whose
# training_metrics.json exists, and two machines with disjoint ARMS write
# disjoint directories.
#
# PPO trials 1-5 on all three envs already exist on the ORIGINATING machine
# from the pilot, at exactly these settings. Use --ignore-existing when
# bringing results home so the pilot's are not overwritten:
#
#     rsync -av --ignore-existing \
#         projects/iclr_2027/runs_actions/gymnax/continual/ \
#         <home>:<repo>/projects/iclr_2027/runs_actions/gymnax/continual/
#
# Then, at home: bash scripts/analysis/finish_iclr_actions.sh
#
# Env: ARMS, ENVS, GPUS, NUM_TRIALS (default 10), JOBS_PER_GPU (default 4).
#
# DO NOT EDIT run_experiments.sh OR launch.sh IN PLACE WHILE THIS RUNS. bash
# reads a script incrementally, so an in-place rewrite under a live launcher
# corrupts its parse. Write a temp file and `mv` it.
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
ROOT=projects/iclr_2027/runs_actions
TRIALS="${NUM_TRIALS:-10}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"

# ------------------------------------------------------------------ preflight
# Refuse to start on a checkout that predates the `actions` branch rather than
# spend 240 GPU-hours on jobs that exit 2 one by one. All four trainers, the
# artifact writer and the evaluator have to know the family; the NE trainers
# additionally have to save the centroid, which every figure in this project now
# reads (runs made before 2026-09-09 10:09 silently plotted the elite instead).
missing=""
for f in train_GA_gymnax_continual train_ES_gymnax_continual \
         train_DNS_gymnax_continual train_RL_gymnax_continual; do
    grep -q "task_type == 'actions'" "source/studies/gymnax/$f.py" \
        || missing="$missing $f:actions"
done
for f in train_GA_gymnax_continual train_ES_gymnax_continual \
         train_DNS_gymnax_continual; do
    grep -q 'centroid=ckpt_centroid' "source/studies/gymnax/$f.py" \
        || missing="$missing $f:centroid"
done
grep -q 'action_flips' source/utils/run_artifacts.py || missing="$missing run_artifacts"
grep -q 'task_type == "actions"' source/studies/evaluate_continual.py \
    || missing="$missing evaluate_continual"
grep -q 'def action_flip_sequence' source/utils/task_sequence.py \
    || missing="$missing task_sequence"
if [ -n "$missing" ]; then
    echo "REFUSING TO LAUNCH: this checkout is missing the action-reversal" >&2
    echo "  and/or centroid support. Missing in:$missing" >&2
    echo "  Pull the 2026-09-09 commits before queueing." >&2
    exit 1
fi

mkdir -p "$ROOT/gymnax"

# ------------------------------------------------------- noncontinual block
# Linked, not retrained -- see the header. A relative symlink so the tree
# survives being rsynced to another path.
if [ ! -e "$ROOT/gymnax/noncontinual" ]; then
    if [ -d "$REPO_ROOT/projects/iclr_2027/runs_centroid/gymnax/noncontinual" ]; then
        ln -s ../../runs_centroid/gymnax/noncontinual "$ROOT/gymnax/noncontinual"
        echo "linked noncontinual -> runs_centroid/gymnax/noncontinual"
    else
        echo "NOTE: runs_centroid/gymnax/noncontinual is not on this machine."
        echo "      Training does not need it; make the symlink at home before"
        echo "      running finish_iclr_actions.sh, which reads it for FT."
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

echo "=== $(date -Is) gymnax ACTION-REVERSAL continual block ==="
echo "    root  : $ROOT"
echo "    arms  : $ARMS"
echo "    envs  : $ENVS"
echo "    gpus  : $GPUS"
echo "    trials: 1..$TRIALS"
echo "    tasks : 20 at period 10, flags 0,1,0,1,... (sub-task 0 = stock order)"

NUM_TRIALS="$TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  ENVS="$ENVS" \
  TASK_TYPE=actions NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 SIGMAS=1.0 \
  LOG_DIR=logs/launch_actions_full \
  bash scripts/train/launch.sh gymnax_continual $ARMS

echo "=== $(date -Is) done ==="
echo "failures:"
grep -hv 'exit=0' logs/launch_actions_full/status.tsv 2>/dev/null || echo "  none"
