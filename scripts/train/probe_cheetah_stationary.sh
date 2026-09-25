#!/bin/bash
# ============================================================================
# THE CHEETAH GATE. Ten stationary runs that decide whether the cheetah block
# can be queued at all, and at which settings. Run this BEFORE
# queue_iclr_cheetah_*.sh exists; nothing downstream is meaningful until it
# has passed.
#
# Every job is a full-budget run of the `cheetah` stationary cell, so a probe
# result is a real number on the reported scale and not a shrunk proxy.
#
# THE RUNS ARE NOT TRIALS OF THAT BLOCK, and an earlier version of this comment
# claimed they were. This script seeds a trial with the trial index (1, 2)
# while `run_condition` seeds it with BASE_SEED + trial - 1 (42, 43), so a
# probe run and a block trial_1 are different runs. They are statistically
# interchangeable -- a stationary cell has no sub-task sequence for a seed to
# select, and bit-reproducibility is not a goal here -- so they can be POOLED
# as extra seeds if trials are ever short. They cannot be dropped into the tree
# as trial_1 and trial_2.
#
# WHAT THIS CAN AND CANNOT DECIDE. Question 2 below is the weak one: PPO
# reaches ~4600 of a ~5000 ceiling within 4% of this budget, so a stationary
# cell saturates and BOTH shapes will finish at the ceiling. That comparison
# belongs to the continual block, where the question is whether a shape
# re-solves a sub-task so fast that no degradation is measurable. Read
# question 2 here as "does the repro2 shape learn at all on the new reward",
# not as a choice between them.
#
# WHY THIS EXISTS
#
# `runs_repro2/mujoco` ran dm_control's CheetahRun through mujoco_playground.
# This runs brax's `halfcheetah` on MJX, because carrying a second simulator
# for one body is the code duplication CLAUDE.md (a) is about. Both are 17-dim
# observation and 6 actuators, so the policy, the searchers and the analysis
# are unchanged -- but the REWARD is not, and that is what makes this a gate
# rather than a formality:
#
#     repro2   dm_control `rewards.tolerance` at _RUN_SPEED 10, ONE-sided
#              (full credit at or above the target), per-step in [0, 1], so an
#              episode caps at 1000.
#     here     TargetSpeedWrapper: a TWO-sided Gaussian peaking AT the target,
#              weight 5, margin 0.5x the target, on a body with no healthy
#              bonus and no termination. Measured 2026-09-09: a random-action
#              cheetah scores ~460 an episode and an untrained network 470-580,
#              against a ceiling of ~5000 (weight 5 x 1000 steps). A standing
#              cheetah would earn 5*exp(-2) = 0.677 a step, so ~640; the
#              measured floor is below that because of the control cost and
#              time spent at negative velocity.
#
# THREE QUESTIONS, and this answers all three in one pass.
#
#   1. IS target_speed 10.0 REACHABLE?  10 is dm_control's constant for
#      dm_control's cheetah xml and `source/envs/mjx.py` flags it PROVISIONAL
#      on this one. If nothing gets near it, every sub-task sits in the same
#      flat low-credit band and the continual block discriminates nothing --
#      which is the failure the ant's 2.0 was checked against before that
#      block was read. Read the final scores against the ~470-580 an untrained
#      network scores: an arm that ends there has learned nothing, one near
#      5000 is at the target, and anything in between says how much of the
#      speed range this body actually covers.
#
#   2. WHICH PPO SHAPE ON THE NEW REWARD?  repro2's cheetah PPO and the shared
#      runner's `_MJX_PPO` agree on gamma 0.97, clip 0.3, GAE 0.95, entropy
#      0.01, grad norm 1 and (128,128) nets, and disagree on three things:
#      learning rate (1e-4 vs 3e-4), update size (81,920 vs 10,240 env steps)
#      and reward scaling (0.1 vs 10). Copying repro2's numbers would match
#      their NAMES rather than their meaning -- `reward_scaling` 0.1 against a
#      reward capped at 1/step is a different object from 0.1 against one
#      capped at 5/step -- so the two shapes are MEASURED here instead.
#      `ppo_repro2` reproduces repro2's update size exactly (4096 x 20 =
#      81,920) at the identical total budget, 6000 updates over the same
#      twenty phases, so the two differ in shape and in nothing else. If both
#      learn, the ant's shape is kept, because one shape across both bodies is
#      worth more than a small win on one of them.
#
#   3. DOES THE SIGMA TRANSFER?  repro2's cheetah GA mutates at 0.1 and the
#      ant's at 0.01. Sigma is a property of the BODY and the policy
#      parameterisation -- a (128,128) tanh MLP over 17 observations and 6
#      actuators -- and neither changed with the simulator, so the cheetah row
#      should still be 0.1. `ga_ant_sigma` is the contrast that says so
#      instead of assuming it. If 0.01 wins, every cheetah NE width needs a
#      sweep before the block is queued.
#
# THE ARMS. Two trials each; ten jobs.
#
#     ppo            the shared runner's shape (lr 3e-4, 512 x 20)
#     ppo_repro2     repro2's shape at the same budget (lr 1e-4, 4096 x 20,
#                    8 epochs, reward_scale 0.1, 6000 updates x 300)
#     es             OpenES at repro2's cheetah row (sigma 0.04, lr 0.01)
#     ga             the GA at repro2's cheetah row (sigma 0.1)
#     ga_ant_sigma   the same GA at the ANT's row (sigma 0.01), the contrast
#
# Each writes to its own directory under the probe root, so the two PPO shapes
# and the two GA widths do not collide, and every override lands in the run's
# config -- a probe run cannot later be mistaken for a reported one.
#
# NOT compute-matched across the PPO pair by accident: both are 4.9152e8
# environment steps over 20 phases, and the CLI prints the effective budget
# and its ratio to the matched one at the top of every job. Read that line.
#
# USAGE
#   DRY_RUN=1 bash scripts/train/probe_cheetah_stationary.sh
#   nohup bash scripts/train/probe_cheetah_stationary.sh \
#       > logs/cheetah_probe.log 2>&1 &
#
#   # read it
#   bash scripts/train/probe_cheetah_stationary.sh --report
#
# COST. The cheetah is the ant's budget on a lighter body, so expect ~1 h a
# trial for the NE arms and ~3 h for PPO, one job to a card. Ten jobs is about
# 16 job-hours.
#
# Env: GPUS (default: whatever is idle), NUM_TRIALS (2), ROOT, ARMS, DRY_RUN.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ROOT="${ROOT:-projects/iclr_2027/probe_cheetah}"
NUM_TRIALS="${NUM_TRIALS:-2}"
DRY_RUN="${DRY_RUN:-0}"
CELL="${CELL:-cheetah}"
ARMS="${ARMS:-ppo ppo_repro2 es ga ga_ant_sigma}"
PY=.venv/bin/python
LOGS=logs/cheetah_probe

# ------------------------------------------------------------------- report
# Reads the finished jobs rather than re-deriving anything: the last record of
# each run's training_metrics.json, which is the FRESH-key centroid score, not
# the search's own selection statistic.
if [ "${1:-}" = "--report" ]; then
    $PY - "$ROOT" <<'PYEOF'
import json, sys, pathlib, statistics
root = pathlib.Path(sys.argv[1])
print(f"{'arm':<14} {'trials':>6} {'final centroid score':>22}   "
      f"untrained 470-580, ceiling ~5000")
for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    vals = []
    for trial in sorted(arm_dir.glob('trial_*')):
        f = trial / 'training_metrics.json'
        if not f.exists():
            continue
        recs = json.loads(f.read_text())
        recs = recs if isinstance(recs, list) else recs.get('records', [])
        if recs:
            vals.append(float(recs[-1].get('centroid_task0',
                                           recs[-1].get('centroid_generalist',
                                                        float('nan')))))
    if vals:
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f'{arm_dir.name:<14} {len(vals):>6} {m:>15.1f} +/- {sd:<6.1f}')
    else:
        print(f'{arm_dir.name:<14} {0:>6} {"(no finished trial)":>22}')
PYEOF
    exit 0
fi

# ------------------------------------------------------------- version guard
if [ ! -f source/studies/mjx/cli.py ]; then
    echo "FATAL: this checkout has no mjx study. Run: git pull" >&2
    exit 1
fi
if ! $PY -c "
import sys; sys.path.insert(0,'.')
from source.studies.mjx import settings as S
assert 'cheetah' in S.CELLS, 'no cheetah cells'
from source.envs import mjx
assert 'friction' in mjx.ENV_CONFIGS['CheetahRun'], 'no cheetah friction grid'
" 2>/dev/null; then
    echo "FATAL: the cheetah cells or its friction grid are missing." >&2
    echo "       This checkout is too old. Run: git pull" >&2
    exit 1
fi

# ------------------------------------------------------------------ the arms
# arm -> extra flags. The METHOD is the first field; everything after it is
# what makes this arm different from the reported settings, and all of it
# lands in the run's config.json.
arm_method() {
    case "$1" in
        ppo|ppo_repro2) echo ppo ;;
        es)             echo es ;;
        ga|ga_ant_sigma) echo ga ;;
        *) echo "unknown arm $1" >&2; exit 1 ;;
    esac
}
arm_flags() {
    case "$1" in
        # repro2's update size (4096 x 20 = 81,920 env steps, exactly its
        # batch_size 256 x unroll 10 x minibatches 32), its lr, its epoch count
        # and its reward scaling -- at the SAME total budget and the same
        # twenty phases, so 6000 updates of 81,920 = 4.9152e8.
        ppo_repro2) echo "--num_updates 6000 --task_interval 300 \
                          --ppo_override num_envs=4096 num_steps=20 \
                          num_epochs=8 learning_rate=0.0001 reward_scale=0.1" ;;
        # The ant's width on the cheetah -- the contrast, not a candidate.
        ga_ant_sigma) echo "--ne_override sigma=0.01" ;;
        *) echo "" ;;
    esac
}

# ------------------------------------------------------------------ GPU pool
if [ -z "${GPUS:-}" ]; then
    # Idle cards only: this shares a machine with whatever else is running, and
    # an MJX job on a busy card is slow enough to look like a failed arm.
    GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | awk -F', ' '$2 < 2000 {printf "%s ", $1}')
    GPUS=${GPUS:-0}
fi
echo "root   : $ROOT"
echo "cell   : $CELL (stationary)"
echo "arms   : $ARMS"
echo "trials : 1..$NUM_TRIALS"
echo "gpus   : $GPUS"
mkdir -p "$LOGS"

# One sequential lane per GPU token, dealt round-robin. No launcher: ten jobs
# do not need one, and this way the probe has no dependency on launch.sh's
# lease file while the gymnax sweeps hold it.
declare -a LANE
n=0
for g in $GPUS; do LANE[$n]=""; n=$((n+1)); done
NLANES=$n
i=0
for arm in $ARMS; do
    method="$(arm_method "$arm")"
    flags="$(arm_flags "$arm")"
    for trial in $(seq 1 "$NUM_TRIALS"); do
        out="$ROOT/$arm/trial_$trial"
        if [ -f "$out/training_metrics.json" ]; then
            echo "SKIP (done) $arm trial $trial"
            continue
        fi
        lane=$((i % NLANES))
        gpu=$(echo $GPUS | cut -d' ' -f$((lane + 1)))
        LANE[$lane]="${LANE[$lane]}
mkdir -p '$out'
echo \">>> \$(date -Is) $arm trial $trial on gpu $gpu\"
$PY source/studies/mjx/cli.py --env $CELL --method $method \
    --trial $trial --seed $trial --gpus $gpu --output_dir '$out' $flags \
    > '$LOGS/${arm}_trial${trial}.log' 2>&1
echo \">>> \$(date -Is) $arm trial $trial exit=\$?\""
        i=$((i + 1))
    done
done

if [ "$i" = "0" ]; then echo "nothing to do -- all jobs already finished"; exit 0; fi
echo "jobs   : $i over $NLANES lane(s)"

if [ "$DRY_RUN" != "0" ]; then
    for l in $(seq 0 $((NLANES - 1))); do
        echo "--- lane $l ---"; echo "${LANE[$l]}"
    done
    echo "=== DRY RUN -- nothing launched ==="
    exit 0
fi

echo "=== $(date -Is) starting ==="
pids=""
for l in $(seq 0 $((NLANES - 1))); do
    [ -z "${LANE[$l]}" ] && continue
    bash -c "${LANE[$l]}" &
    pids="$pids $!"
done
wait $pids
echo "=== $(date -Is) done ==="
bash "$0" --report
