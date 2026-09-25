#!/bin/bash
# ============================================================================
# Repair the (budget, task_period) mismatch in
# projects/iclr_2027/runs_centroid/gymnax/continual.
#
# WHAT WENT WRONG. block_gymnax_continual defaults to NUM_TASKS=10 and
# TASK_PERIOD=0. Two single-arm queues set only PROJECT_ROOT / NUM_TRIALS /
# SIGMAS and inherited those defaults, so they trained HALF the budget over ten
# sub-tasks that are never revisited, into cells whose other arms are at
# 3.072e9 env steps over twenty sub-tasks seen twice:
#
#   logs/launch_centroid_trac_retry      trac, 10 trials across all three envs
#   queue_iclr_centroid_dns_gaussian.sh  dns_gaussian, all 20 CartPole and
#                                        Acrobot trials, MountainCar in flight
#
# Both are unusable twice over: CLAUDE.md rule (c) (the arms in a cell no
# longer saw the same number of steps, nor switched at the same step) and,
# independently, TASK_PERIOD=0 revisits nothing, so forgetting -- the drop on a
# sub-task you return to -- is not defined on them at all.
#
# The root cause is fixed in both queue scripts; this only cleans up after it.
#
# WHAT THIS DOES
#   1. stop   the dns_gaussian continual launcher and its trainers, which are
#             still producing more of them. Matched on the EXACT single-method
#             argv and on a `_sigma1.0` output directory, so the concurrent
#             queue_iclr_centroid_mcar_sigma0.1.sh launcher -- whose own
#             dns_gaussian jobs write to `_sigma0.1` and ARE correct -- is not
#             touched.
#   2. list   every COMPLETED continual run whose (steps, task_period) differs
#             from its cell's majority. Derived from the configs on disk, not
#             from a hardcoded list: a hardcoded list is how the wrong trial
#             gets deleted, and this is a delete.
#   3. delete those trial directories, after printing them.
#   4. queue  trac and dns_gaussian again at the cell's settings. Deletion has
#             to come first: run_condition skips a trial whose
#             training_metrics.json exists, so a re-queue over the bad runs
#             would skip them and change nothing.
#
# RUN THE DRY RUN FIRST. It does step 2 only -- nothing killed, deleted or
# queued -- and prints exactly what the real run would remove:
#
#   DRY_RUN=1 bash scripts/analysis/fix_centroid_budget_mismatch.sh
#   bash scripts/analysis/fix_centroid_budget_mismatch.sh
#
# GPUs: leaves 4 and 5 to the sigma0.1 sweep; the re-queues take 0-3, 6 and 7.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY=.venv/bin/python
ROOT=projects/iclr_2027/runs_centroid/gymnax/continual
DRY_RUN="${DRY_RUN:-0}"
LIST=/tmp/centroid_mismatch.txt

# ---------------------------------------------------------------- 1. stop
if [ "$DRY_RUN" = "0" ]; then
    echo "=== 1/4 stopping the dns_gaussian continual launcher ==="
    # The launcher loop and its per-job subshells share this argv. Matching the
    # method list exactly -- 'dns_gaussian' and nothing after it -- is what
    # keeps the sigma0.1 launcher, whose argv lists eight methods, out of it.
    L=$(pgrep -f 'launch\.sh gymnax_continual dns_gaussian$' | tr '\n' ' ')
    # Trainers are matched on the OUTPUT DIRECTORY, the only thing separating
    # the wrong-settings sigma1.0 jobs from the correct sigma0.1 ones running
    # beside them under the other launcher.
    T=$(pgrep -f "output_dir $ROOT/dns_gaussian/[A-Za-z_0-9]*_sigma1\.0" | tr '\n' ' ')
    echo "  launchers: ${L:-none}"
    echo "  trainers : ${T:-none}"
    [ -n "${L// /}" ] && kill $L 2>/dev/null
    [ -n "${T// /}" ] && kill $T 2>/dev/null
    sleep 8
    # SIGKILL only what ignored SIGTERM; a JAX process inside a compile can.
    L=$(pgrep -f 'launch\.sh gymnax_continual dns_gaussian$' | tr '\n' ' ')
    T=$(pgrep -f "output_dir $ROOT/dns_gaussian/[A-Za-z_0-9]*_sigma1\.0" | tr '\n' ' ')
    [ -n "${L// /}" ] && kill -9 $L 2>/dev/null
    [ -n "${T// /}" ] && kill -9 $T 2>/dev/null
    sleep 2
    echo "  sigma1.0 dns_gaussian trainers left (want 0): $(pgrep -cf "output_dir $ROOT/dns_gaussian/[A-Za-z_0-9]*_sigma1\.0" || true)"
    echo "  sigma0.1 trainers still alive (want > 0)    : $(pgrep -cf '_sigma0\.1/trial_' || true)"
fi

# ------------------------------------------------------- 2. list, 3. delete
echo "=== 2/4 completed runs whose (steps, task_period) differ from their cell ==="
$PY - "$ROOT" > "$LIST" <<'PYEOF'
import collections, json, pathlib, sys

root = pathlib.Path(sys.argv[1])
sig = {}
for method in sorted(p for p in root.iterdir() if p.is_dir()):
    for cell in sorted(p for p in method.iterdir() if p.is_dir()):
        for trial in sorted(cell.glob('trial_*')):
            # COMPLETED runs only. A trial still training has no
            # training_metrics.json, and must not be deleted out from under a
            # live trainer -- including the sigma0.1 sweep's own jobs.
            if not (trial / 'training_metrics.json').exists():
                continue
            results = trial / 'results.json'
            if not results.exists():
                continue
            try:
                config = json.load(open(results)).get('config', {})
            except Exception:
                continue
            gens = config.get('num_generations')
            steps = config.get('num_timesteps')
            # One currency for both families: NE records generations, RL
            # records env steps, and a generation is pop x evals x ep_length.
            if steps is None:
                steps = (gens or 0) * 512 * 3 * 500
            sig[trial] = (steps, config.get('task_period'))

by_cell = collections.defaultdict(list)
for trial, s in sig.items():
    by_cell[trial.parent.name].append((trial, s))

for cell, entries in sorted(by_cell.items()):
    majority = collections.Counter(s for _, s in entries).most_common(1)[0][0]
    for trial, s in sorted(entries):
        if s != majority:
            print(trial)
PYEOF

n=$(wc -l < "$LIST")
echo "  $n trial directories:"
sed 's/^/    /' "$LIST"

if [ "$DRY_RUN" = "1" ]; then
    echo "=== DRY_RUN: stopping here. Nothing killed, deleted or queued. ==="
    exit 0
fi

echo "=== 3/4 deleting them ==="
if [ "$n" -gt 0 ]; then
    # Guard: every path must be a trial_* directory inside this run tree.
    while read -r d; do
        case "$d" in
            "$ROOT"/*/*/trial_*)
                rm -rf "$d"
                echo "    removed $d"
                ;;
            *)
                echo "    REFUSED (not a trial dir under $ROOT): $d"
                ;;
        esac
    done < "$LIST"
else
    echo "    nothing to delete"
fi

# --------------------------------------------------------------- 4. requeue
echo "=== 4/4 re-queueing trac and dns_gaussian at the cell's settings ==="
export GPUS="${GPUS:-0 0 1 1 2 2 3 3 6 6 7 7}"
bash scripts/train/queue_iclr_centroid_trac.sh
bash scripts/train/queue_iclr_centroid_dns_gaussian.sh

echo "=== done; re-audit with scripts/verify_runs.py ==="
