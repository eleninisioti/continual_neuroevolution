#!/bin/bash
# ============================================================================
# THE CLUSTER'S HALF of the ICLR 2027 gymnax grid. Self-contained: it runs an
# EXPLICIT list of 152 trials and nothing else, so it needs no copy of the home
# machine's run tree and cannot duplicate work already done there.
#
# THE GRID. Three cells, eight arms, ten trials, noise chosen per environment:
#
#     CartPole-v1     sigma 1.0
#     Acrobot-v1      sigma 1.0
#     MountainCar-v0  sigma 0.1    <- not 1.0, which is 7-15x its velocity range
#     arms  ga es nes dns_gaussian ppo trac redo cchain
#
# `ga` and `dns_gaussian` are the GAUSSIAN pair, both at --mutation_std 0.5, so
# they differ in the selection rule alone. The Iso+LineDD arms (`ga_isoline`,
# `dns`) are not in the paper and are not run here.
#
# WHAT IS NOT IN THE LIST, and why:
#
#   74 trials  ppo / redo / cchain / trac on CartPole and Acrobot sigma 1.0.
#              Already finished on the home machine. The RL trainers save
#              `final`, the deployed policy, which is what the centroid
#              plasticity figure reads for them, so they never needed redoing.
#   14 trials  ga MountainCar 9-10, dns_gaussian MountainCar 1-6,
#              trac CartPole 4/6/10, trac Acrobot 4/6/8 -- in flight at home
#              when this list was cut (2026-09-09 11:20). That is why `ga`
#              MountainCar is trials 1-8 and `dns_gaussian` is 7-10.
#
# WHY THE NE ARMS ARE RE-RUN AT ALL. The plasticity figure under
# `--agent centroid` needs `ne_centroid_*` and `checkpoints.npz['centroid']`,
# which no trainer wrote before 2026-09-09 10:09. Without them it silently
# falls back to the ELITE. The lineplot is unaffected. Nothing recovers it post
# hoc, so the runs have to be made again.
#
# USAGE
#   git pull       # must include affa339 or the guard below stops you
#   DRY_RUN=1 bash scripts/train/queue_iclr_centroid_cluster.sh   # print the list
#   nohup bash scripts/train/queue_iclr_centroid_cluster.sh > logs/cluster.log 2>&1 &
#
#   # when it is done, send the runs to the home machine
#   rsync -av --ignore-existing \
#       projects/iclr_2027/runs_centroid/gymnax/continual/ \
#       <home>:<repo>/projects/iclr_2027/runs_centroid/gymnax/continual/
#
# The two blocks run CONCURRENTLY on disjoint halves of the GPU pool, because
# launch.sh leases one token per GPUS entry and knows nothing about a second
# launcher -- two launchers sharing a pool would each claim all of it. The
# halves are 6,820 and 6,810 job-minutes, so they finish together.
#
# PART splits the 152 between the two machines, disjointly BY CONSTRUCTION --
# the two parts are complements of one line-level predicate, so no trial can
# land in both and none can fall through the gap:
#
#   PART=home     the cheap MountainCar sigma 0.1 arms -- ga, dns_gaussian,
#                 es, ppo, redo. 42 trials, 3,100 job-minutes, which is about
#                 100 minutes on 8 cards at 4 jobs each.
#   PART=cluster  everything else: both sigma 1.0 environments entire, plus
#                 nes / trac / cchain on MountainCar. 110 trials, 11,050
#                 job-minutes. The default.
#   PART=all      all 152, for one machine doing the lot.
#
# Run `PART=home` here and `PART=cluster` there and the union is exactly the
# grid, counted once.
#
# Env: PART (cluster), GPUS, JOBS_PER_GPU (4), DRY_RUN, ROOT.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ROOT="${ROOT:-projects/iclr_2027/runs_centroid}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"
DRY_RUN="${DRY_RUN:-0}"
PART="${PART:-cluster}"
LOGS=logs/cluster_centroid
mkdir -p "$LOGS"

# ------------------------------------------------------------ version guard
# This is the check whose absence cost eight GA trials on 2026-09-09: a
# launcher started at 09:18 against a trainer edited at 10:09, and Python reads
# the source at process start, so runs that FINISHED after the edit still had
# no centroid.
for f in ES GA DNS; do
    t="source/studies/gymnax/train_${f}_gymnax_continual.py"
    if ! grep -q 'ckpt_centroid' "$t" 2>/dev/null; then
        echo "FATAL: $t cannot write the centroid checkpoint." >&2
        echo "       This checkout predates affa339. Run: git pull" >&2
        exit 1
    fi
done
echo "version guard OK: trainers write ne_centroid_* and checkpoints.npz[centroid]"

# ------------------------------------------------------------ the job lists
# One list per sigma, because SIGMAS is a per-block setting.
S10="$LOGS/jobs_sigma1.0.tsv"
S01="$LOGS/jobs_sigma0.1.tsv"

: > "$S10"
for method in ga es nes dns_gaussian; do
    for env in CartPole-v1 Acrobot-v1; do
        for trial in $(seq 1 10); do
            printf '%s\t%s\t%s\n' "$method" "$env" "$trial" >> "$S10"
        done
    done
done

: > "$S01"
for method in es nes ppo trac redo cchain; do
    for trial in $(seq 1 10); do
        printf '%s\tMountainCar-v0\t%s\n' "$method" "$trial" >> "$S01"
    done
done
for trial in 1 2 3 4 5 6 7 8;  do printf 'ga\tMountainCar-v0\t%s\n' "$trial" >> "$S01"; done
for trial in 7 8 9 10;         do printf 'dns_gaussian\tMountainCar-v0\t%s\n' "$trial" >> "$S01"; done

# The partition. `home` is MountainCar sigma 0.1 for the five cheap arms;
# `cluster` is the complement. Applied to the built lists rather than to the
# generators above, so the two parts cannot drift apart as the grid changes.
HOME_RE='^(ga|dns_gaussian|es|ppo|redo)\tMountainCar-v0\t'
case "$PART" in
    home)
        grep -P "$HOME_RE" "$S01" > "$S01.part" && mv "$S01.part" "$S01"
        : > "$S10" ;;
    cluster)
        grep -Pv "$HOME_RE" "$S01" > "$S01.part" && mv "$S01.part" "$S01" ;;
    all) ;;
    *) echo "FATAL: PART must be home, cluster or all (got '$PART')" >&2; exit 1 ;;
esac
echo "part            : $PART"
echo "sigma 1.0 block : $(grep -c . "$S10") trials  ($S10)"
echo "sigma 0.1 block : $(grep -c . "$S01") trials  ($S01)"
echo "total           : $(( $(grep -c . "$S10") + $(grep -c . "$S01") ))"

if [ "$DRY_RUN" != "0" ]; then
    echo "--- sigma 1.0 ---"; cat "$S10"
    echo "--- sigma 0.1 ---"; cat "$S01"
    echo "=== DRY RUN -- nothing queued ==="
    exit 0
fi

# ------------------------------------------------------------ GPU split
if [ -z "${GPUS:-}" ]; then
    VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')
    VISIBLE=${VISIBLE:-0}
    GPUS=""
    for g in $VISIBLE; do
        for _ in $(seq 1 "$JOBS_PER_GPU"); do GPUS="$GPUS $g"; done
    done
fi
read -r -a POOL <<< "$(echo $GPUS)"
HALF=$(( ${#POOL[@]} / 2 ))
[ "$HALF" -lt 1 ] && HALF=1
GPUS_A="${POOL[*]:0:$HALF}"
GPUS_B="${POOL[*]:$HALF}"
[ -z "$GPUS_B" ] && GPUS_B="$GPUS_A"
if [ ! -s "$S10" ]; then GPUS_B="${POOL[*]}"; GPUS_A=""; fi
if [ ! -s "$S01" ]; then GPUS_A="${POOL[*]}"; GPUS_B=""; fi
echo "sigma 1.0 gpus  : $GPUS_A"
echo "sigma 0.1 gpus  : $GPUS_B"

# ------------------------------------------------------------ run
echo "=== $(date -Is) starting both blocks ==="
if [ -s "$S10" ]; then
GPUS="$GPUS_A" JOBLIST="$S10" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 SIGMAS=1.0 \
  LOG_DIR="$LOGS/sigma1.0" \
  bash scripts/train/launch.sh gymnax_continual ignored &
PID_A=$!
else
PID_A=""
fi

GPUS="$GPUS_B" JOBLIST="$S01" PROJECT_ROOT="$ROOT" TRACK_DIVERSITY=1 \
  NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200 SIGMAS=0.1 \
  LOG_DIR="$LOGS/sigma0.1" \
  bash scripts/train/launch.sh gymnax_continual ignored &
PID_B=$!

A=0; [ -n "$PID_A" ] && { wait $PID_A; A=$?; }
wait $PID_B; B=$?
echo "=== $(date -Is) done (sigma1.0 exit=$A, sigma0.1 exit=$B) ==="
echo "failures:"
grep -hv 'exit=0' "$LOGS"/*/status.tsv 2>/dev/null || echo "  none"
