#!/bin/bash
# ============================================================================
# Spread one experiment block over every GPU on the box.
#
#   scripts/train/launch.sh <block> <methods...>
#
#   BLOCK is anything run_experiments.sh accepts (gymnax_noncontinual,
#   gymnax_continual, mujoco_continual, brax_continual, ...).
#
# Why a launcher rather than `run_experiments.sh <block> all`: that script
# walks methods x envs x trials in one process on one GPU, which on the gymnax
# blocks is ~120 GPU-hours in series. The unit of work here is a single
# (method, env, trial) -- the finest granularity that is still one process --
# and one runs per GPU at a time, each leasing a GPU for its duration.
#
# ONE JOB PER GPU IF THE RUN MUST BE BIT-REPRODUCIBLE
# ---------------------------------------------------
# Two trainers sharing a GPU change the RL results. Measured on
# gymnax_noncontinual ppo, CartPole, seed 42: run alone, it reproduces the
# reported trial update for update (202.00 +- 107.64, 191.10 +- 55.28,
# 356.00 +- 75.59); run as one of 14 concurrent jobs, the same command gives
# 176.20 / 165.50 / 297.00 and forks from there. Nothing about the command
# differs -- under memory pressure XLA autotunes to different kernels, and the
# reduction order inside them is not the same.
#
# The NE methods are NOT affected: GA and ES were bit-identical over 600
# generations x 30 trials at the same concurrency, because their per-generation
# work is small vector ops rather than the RL value net's 3x128 matmuls.
#
# JOBS=<number of GPUs> DID NOT DELIVER THAT, AND THAT IS FIXED HERE
# -------------------------------------------------------------------
# Until 2026-08-06 this script pre-assigned each job a GPU by its *index* in the
# job list (`gpu=${GPU_ARR[i % n]}`) and handed the list to `xargs -P`. xargs
# starts the next line whenever ANY slot frees, so as soon as jobs stopped
# finishing in list order the running set no longer occupied distinct GPUs: job
# k+8 started on GPU (k mod 8) while job k was still on it, and a different GPU
# sat idle. Caught in the act during the gymnax reproduction, with JOBS=8 on 8
# GPUs -- the configuration that was supposed to make this impossible:
#
#     GPU 3: 1181 MiB (two trainers)      GPU 5: 15 MiB (idle)
#
# and gymnax_noncontinual ppo CartPole trial 1 came back at 176.00 / 165.40 /
# 296.80 -- the contended signature above, to within 0.2 -- while the reported
# run matches "alone" exactly. Only 10 of 30 PPO trials reproduced; GA and ES
# were unaffected at 30/30, exactly as the note above predicts for NE.
#
# A GPU is now a LEASE, not a label: one token per GPU in a FIFO, a job pops one
# and pushes it back on exit. A GPU cannot host two jobs whatever order things
# finish in, and no GPU idles while work is queued. Concurrency is therefore
# always exactly the number of GPUs, and there is no JOBS knob any more --
# setting it above the GPU count was the bug. To deliberately oversubscribe (new
# runs where only statistics matter), repeat GPUs: GPUS="0 0 1 1".
#
# Environment:
#   GPUS="0 1 2 3"     GPUs to use            (default: all visible)
#   NUM_TRIALS=10      trials per condition
#   ENVS="..."         override the block's env list
#   PROJECT_ROOT=...   where runs land; point this at a scratch tree to
#                      reproduce without touching the reported one
#   LOG_DIR=...        per-job logs (default: logs/launch)
#   DRY_RUN=1          print the job list and exit
#
# Each job appends one line to $LOG_DIR/status.tsv on exit, so a failed trial is
# visible without reading 400 logs:
#
#   grep -v 'exit=0' $LOG_DIR/status.tsv
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Walk up to the directory holding pyproject.toml rather than counting
# `dirname`s: every time these scripts moved between `scripts/` and
# `scripts/outdated/` a hardcoded count went stale, and the failure was 60 jobs
# exiting 127 before anything trained.
REPO_ROOT="$SCRIPT_DIR"
while [ ! -f "$REPO_ROOT/pyproject.toml" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
cd "$REPO_ROOT"

RUNNER="$REPO_ROOT/scripts/train/run_experiments.sh"

BLOCK="${1:-}"
shift || true
METHODS="$*"
if [ -z "$BLOCK" ] || [ -z "$METHODS" ]; then
    sed -n '2,30p' "$0"
    exit 1
fi

if [ -z "${GPUS:-}" ]; then
    GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')
    GPUS=${GPUS:-0}
fi
GPU_ARR=($GPUS)
NUM_TRIALS="${NUM_TRIALS:-10}"
PROJECT_ROOT="${PROJECT_ROOT:-projects/neurips_2026_rebuttal/runs}"
LOG_DIR="${LOG_DIR:-logs/launch}"
ENVS="${ENVS:-CartPole-v1 Acrobot-v1 MountainCar-v0}"

mkdir -p "$LOG_DIR"
STATUS="$LOG_DIR/status.tsv"

# The job list: one line per (method, env, trial). No GPU column -- the GPU is
# leased at run time, which is the whole point (see the header).
#
# ONE JOBFILE PER INVOCATION, NOT ONE PER LOG DIRECTORY
# ----------------------------------------------------
# This used to be a fixed `$LOG_DIR/jobs.txt`, truncated on every launch. The
# job loop at the bottom reads that path with `while read ... < "$JOBFILE"`, so
# a second launch sharing the log directory rewrote the file OUT FROM UNDER the
# first loop's open descriptor, and the first launch carried on executing the
# second one's jobs.
#
# It happened, and it cost the gymnax NE reproduction: `queue_ne_only.sh`
# announced `gymnax_continual ga dns es` (90 jobs) at 13:36 on 2026-08-06, an
# RL launch overwrote jobs.txt with 120 ppo/trac/redo/cchain jobs at 14:34, and
# the surviving loop spent the next day running cchain into `runs_repro` while
# reporting itself as the ga/dns/es launch. `runs_repro/gymnax/continual` came
# out with 90/90 trac and redo, 68 cchain -- and zero dns, zero es.
#
# $$ is the shell's PID, so concurrent launches cannot collide whatever log
# directory they share. The file is left behind on purpose: it is the record of
# what a launch actually queued, which is exactly what was missing above.
JOBFILE="$LOG_DIR/jobs.$$.txt"
: > "$JOBFILE"
i=0
# JOBLIST is an EXPLICIT job list -- one `method<TAB>env<TAB>trial` per line --
# used instead of the METHODS x ENVS x 1..NUM_TRIALS cross product. It exists
# because the cross product cannot say "ga trials 1-8 but dns_gaussian 7-10",
# which is what splitting a half-finished cell across two machines needs. The
# alternative was to rely on run_condition skipping trials already on disk,
# and that only works if the two machines share a filesystem or you copy the
# runs across first. METHODS/ENVS are ignored when it is set; SIGMAS and the
# rest of the block settings still apply, so one list per sigma.
if [ -n "${JOBLIST:-}" ]; then
    if [ ! -f "$JOBLIST" ]; then
        echo "FATAL: JOBLIST=$JOBLIST does not exist" >&2; exit 1
    fi
    grep -v '^[[:space:]]*$' "$JOBLIST" > "$JOBFILE"
    i=$(grep -c . "$JOBFILE")
else
    for method in $METHODS; do
        for env in $ENVS; do
            for trial in $(seq 1 "$NUM_TRIALS"); do
                echo -e "${method}\t${env}\t${trial}" >> "$JOBFILE"
                i=$((i + 1))
            done
        done
    done
fi

echo "=========================================="
echo "block      : $BLOCK"
if [ -n "${JOBLIST:-}" ]; then
    echo "methods    : $(cut -f1 "$JOBFILE" | sort -u | tr '\n' ' ')"
else
    echo "methods    : $METHODS"
fi
echo "envs       : $ENVS"
if [ -n "${JOBLIST:-}" ]; then
    echo "trials     : explicit list, $JOBLIST"
else
    echo "trials     : 1..$NUM_TRIALS"
fi
echo "jobs       : $i, one per GPU at a time over [$GPUS] (leased, not pre-assigned)"
echo "runs ->    : $PROJECT_ROOT"
echo "logs ->    : $LOG_DIR"
echo "=========================================="

if [ "${DRY_RUN:-0}" = "1" ]; then
    cat "$JOBFILE"
    exit 0
fi

# One token per GPU. `read` from the FIFO blocks until a GPU is genuinely free.
POOL_DIR="$(mktemp -d)"
mkfifo "$POOL_DIR/pool"
exec 9<>"$POOL_DIR/pool"          # read+write, so opening never blocks
rm -rf "$POOL_DIR"
for g in "${GPU_ARR[@]}"; do echo "$g" >&9; done
trap 'exec 9>&- 2>/dev/null || true' EXIT

pids=()
while IFS=$'\t' read -r method env trial; do
    read -r gpu <&9                            # waits for a free GPU
    (
        tag="${BLOCK}.${method}.${env}.trial${trial}"
        log="$LOG_DIR/${tag}.log"
        GPU="$gpu" TRIALS="$trial" ENVS="$env" PROJECT_ROOT="$PROJECT_ROOT" \
            bash "$RUNNER" "$BLOCK" "$method" > "$log" 2>&1
        code=$?
        printf '%s\texit=%d\tgpu=%s\t%s\n' "$tag" "$code" "$gpu" "$log" >> "$STATUS"
        [ "$code" = "0" ] || echo "!! FAILED $tag (see $log)"
        echo "$gpu" >&9                        # hand it back, success or not
    ) &
    pids+=($!)
done < "$JOBFILE"

for p in "${pids[@]}"; do wait "$p"; done

echo
echo "done. failures:"
grep -v 'exit=0' "$STATUS" 2>/dev/null || echo "  none"
