#!/bin/bash
# ============================================================================
# Bring projects/iclr_2027/runs_centroid to the state BOTH ICLR gymnax figures
# need: the training lineplot + its table, and the centroid plasticity figure.
#
# THE TARGET. Three cells, eight arms, ten trials, observation noise chosen per
# environment rather than globally:
#
#     CartPole-v1      sigma 1.0
#     Acrobot-v1       sigma 1.0
#     MountainCar-v0   sigma 0.1     <- NOT 1.0. Measured 2026-09-09: sigma 1.0
#                                      is 7-15x MountainCar's velocity range, so
#                                      the observation is noise. Every
#                                      MountainCar_v0_sigma1.0 cell is outdated.
#
#     arms   ga es nes dns_gaussian ppo trac redo cchain
#
# `ga_isoline` and `dns` are the Iso+LineDD pair, dropped from the paper on
# 2026-09-09 (the reported comparison is operator-matched: gaussian `ga` vs
# gaussian `dns_gaussian`), so they are outdated too.
#
# WHY THE NE ARMS MUST BE RE-RUN. The plasticity figure under `--agent centroid`
# needs two things no trainer wrote before 10:09 on 2026-09-09:
#
#     ne_centroid_*             per-generation dormancy / action-churn / NTK of
#                               the coordinate-wise mean of the population's
#                               weights, beside the ne_elite_* twins.
#     checkpoints.npz[centroid] that same vector saved once per sub-task.
#
# Without them the figure falls back to the ELITE -- silently for the curve
# rows, and via `incumbent` for the checkpoint rows, which IS the centroid for
# ES/NES (`es_state.mean`) but is `archive[0]` for the GA and the repertoire
# argmax for DNS. Nothing recovers it post hoc: `mean_checkpoints/task_*.pkl`
# is `mean_params`, byte identical to `incumbent`, and the per-generation
# columns need the population at every generation. The LINEPLOT is unaffected
# -- `centroid_fitness` has been logged since 2026-09-08 and is in every trial
# -- so this re-run is entirely for the second figure.
#
# The RL arms (ppo trac redo cchain) are NOT affected: they save `final`, the
# deployed policy, and log `policy_*` on that same network, which is what
# `--agent centroid` resolves to for them. They are re-run here only where the
# MountainCar sigma 0.1 cell is empty.
#
# WHAT IT DOES
#   1. list   every trial that is outdated -- a cell outside the target, an arm
#              outside the list, or an NE trial in a target cell whose
#              checkpoints.npz has no `centroid` key. Derived from the files on
#              disk, never a hardcoded trial list. Trials with no
#              training_metrics.json are IN FLIGHT and are left alone.
#   2. move   them to $ASIDE (default projects/iclr_2027/runs_outdated),
#              preserving the tree shape. A move, not a delete: the lineplot
#              data in them is still valid and this stays reversible.
#   3. queue  the three target cells. Every setting is spelled out --
#              block_gymnax_continual defaults to NUM_TASKS=10 / TASK_PERIOD=0
#              and a queue that sets only PROJECT_ROOT has now twice written
#              half-budget runs into a full-budget cell (CLAUDE.md rule (c):
#              every arm in a cell sees the same steps and switches at the same
#              step). run_condition skips any trial already on disk, so this is
#              safe to re-run and safe to run on two machines at once.
#
# ON THE CLUSTER
#   git pull                      # needs affa339 or later, or step 3 writes
#                                 # runs with no centroid and you repeat this
#   DRY_RUN=1 bash scripts/train/queue_iclr_centroid_target.sh
#   nohup bash scripts/train/queue_iclr_centroid_target.sh \
#       > logs/centroid_target.log 2>&1 &
#
#   # then bring it home
#   rsync -av --ignore-existing \
#       projects/iclr_2027/runs_centroid/gymnax/continual/ \
#       <home>:<repo>/projects/iclr_2027/runs_centroid/gymnax/continual/
#
# THEN, at home, rebuild both figures and CHECK THE AGENT:
#   bash scripts/analysis/finish_iclr_centroid.sh
#   grep -o '"agent_source": "[a-z]*"' \
#     projects/iclr_2027/results/gymnax_centroid/centroid/sigma*/plasticity_checkpoints.json \
#     | sort -u        # must be "centroid" for every NE arm, never "incumbent"
#
# Env: ARMS, NUM_TRIALS (10), TRIALS (a SUBSET of indices, e.g. "1 2 3 4 5",
#      for splitting one cell across machines), JOBS_PER_GPU (4), GPUS,
#      DRY_RUN, ASIDE, ROOT,
#      SKIP_MOVE, SKIP_QUEUE, CELLS ("cartpole acrobot mountaincar").
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY=.venv/bin/python
ROOT="${ROOT:-projects/iclr_2027/runs_centroid}"
CONT="$ROOT/gymnax/continual"
ARMS="${ARMS:-ga es nes dns_gaussian ppo trac redo cchain}"
NUM_TRIALS="${NUM_TRIALS:-10}"
# run_experiments.sh:177 reads TRIALS as the LIST of trial indices to run
# ("1 2 3 4 5"), which is how one cell is split across machines. It is NOT
# the count -- that is NUM_TRIALS. Passed through untouched; unset it means
# every trial. Naming the count TRIALS here would have silently run trial 10
# and nothing else.
TRIALS="${TRIALS:-}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"
DRY_RUN="${DRY_RUN:-0}"
ASIDE="${ASIDE:-projects/iclr_2027/runs_outdated}"
SKIP_MOVE="${SKIP_MOVE:-0}"
SKIP_QUEUE="${SKIP_QUEUE:-0}"
CELLS="${CELLS:-cartpole acrobot mountaincar}"
LIST=/tmp/centroid_outdated.txt

# The target, in one place, read by both the mover and the queue below.
TARGET_CELLS="CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma0.1"
NE_ARMS="ga es nes dns_gaussian"

# ------------------------------------------------------- 0. version guard
# The whole point of this script is the centroid columns. Refuse to queue
# anything from a checkout that cannot write them -- that is exactly how the
# eight GA MountainCar trials that finished at 10:24 on 2026-09-09 came out
# unusable: the launcher started before the trainer was edited.
for f in ES GA DNS; do
    t="source/studies/gymnax/train_${f}_gymnax_continual.py"
    if ! grep -q 'ckpt_centroid' "$t" 2>/dev/null; then
        echo "FATAL: $t does not save the centroid checkpoint." >&2
        echo "       This checkout predates affa339 (2026-09-09). git pull first," >&2
        echo "       or every NE run this queues is outdated on arrival." >&2
        exit 1
    fi
done
echo "version guard: trainers write ne_centroid_* and checkpoints.npz[centroid]"

# ------------------------------------------------------- 1. list outdated
echo "=== 1/3 outdated trials in $CONT ==="
CONT="$CONT" TARGET_CELLS="$TARGET_CELLS" ARMS="$ARMS" NE_ARMS="$NE_ARMS" \
"$PY" - <<'EOF' | tee "$LIST"
import os, sys, numpy as np
cont = os.environ['CONT']
target = set(os.environ['TARGET_CELLS'].split())
arms = set(os.environ['ARMS'].split())
ne = set(os.environ['NE_ARMS'].split())
if not os.path.isdir(cont):
    sys.exit(0)
for arm in sorted(os.listdir(cont)):
    ad = os.path.join(cont, arm)
    if not os.path.isdir(ad):
        continue
    for cell in sorted(os.listdir(ad)):
        cd = os.path.join(ad, cell)
        if not os.path.isdir(cd):
            continue
        for trial in sorted(os.listdir(cd)):
            t = os.path.join(cd, trial)
            if not os.path.exists(os.path.join(t, 'training_metrics.json')):
                continue                      # in flight -- leave it alone
            if arm not in arms:
                print(t, '# arm dropped from the paper'); continue
            if cell not in target:
                print(t, '# cell is not a target cell'); continue
            if arm in ne:
                z = os.path.join(t, 'checkpoints.npz')
                try:
                    ok = 'centroid' in np.load(z, allow_pickle=True).files
                except Exception:
                    ok = False
                if not ok:
                    print(t, '# NE run with no centroid checkpoint')
EOF
N=$(grep -c . "$LIST" || true)
echo "    $N outdated trial(s)"

if [ "$DRY_RUN" != "0" ]; then
    echo "=== DRY RUN -- nothing moved, nothing queued ==="
    exit 0
fi

# ------------------------------------------------------- 2. move aside
if [ "$SKIP_MOVE" = "0" ] && [ "$N" != "0" ]; then
    echo "=== 2/3 moving $N trial(s) to $ASIDE ==="
    while read -r t _rest; do
        [ -n "$t" ] || continue
        dst="$ASIDE/${t#$ROOT/}"
        mkdir -p "$(dirname "$dst")"
        rm -rf "$dst"
        mv "$t" "$dst" && echo "    $t -> $dst"
    done < "$LIST"
    find "$CONT" -mindepth 2 -maxdepth 2 -type d -empty -delete 2>/dev/null
    find "$CONT" -mindepth 1 -maxdepth 1 -type d -empty -delete 2>/dev/null
else
    echo "=== 2/3 skipped ==="
fi

[ "$SKIP_QUEUE" = "1" ] && { echo "=== 3/3 skipped ==="; exit 0; }

# ------------------------------------------------------- 3. queue
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
echo "=== 3/3 $(date -Is) queueing ==="
echo "    arms   : $ARMS"
echo "    trials : ${TRIALS:-1..$NUM_TRIALS}"
echo "    gpus   : $GPUS"

# The two sigma-1.0 environments. One launch.sh call so the arms in these two
# cells interleave over the same GPU pool rather than one arm hogging it.
case " $CELLS " in *" cartpole "*|*" acrobot "*)
    ENVS_10=""
    case " $CELLS " in *" cartpole "*) ENVS_10="CartPole-v1";; esac
    case " $CELLS " in *" acrobot "*) ENVS_10="$ENVS_10 Acrobot-v1";; esac
    echo "=== $(date -Is) sigma 1.0 block:$ENVS_10 ==="
    NUM_TRIALS="$NUM_TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
      ENVS="${ENVS_10# }" \
      NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 SIGMAS=1.0 \
      LOG_DIR=logs/launch_centroid_target_sigma1.0 \
      bash scripts/train/launch.sh gymnax_continual $ARMS
;; esac

# MountainCar at its own sigma. A separate call because SIGMAS is per-block.
case " $CELLS " in *" mountaincar "*)
    echo "=== $(date -Is) MountainCar sigma 0.1 block ==="
    NUM_TRIALS="$NUM_TRIALS" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
      ENVS="MountainCar-v0" \
      NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 SIGMAS=0.1 \
      LOG_DIR=logs/launch_centroid_target_mcar_sigma0.1 \
      bash scripts/train/launch.sh gymnax_continual $ARMS
;; esac

echo "=== $(date -Is) done ==="
echo "failures:"
grep -hv 'exit=0' logs/launch_centroid_target_*/status.tsv 2>/dev/null || echo "  none"
