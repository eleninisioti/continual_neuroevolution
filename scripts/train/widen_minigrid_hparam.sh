#!/bin/bash
# ============================================================================
# Re-run the MiniGrid stage of resume_hparam_sigma.sh at full width.
#
#     nohup bash scripts/train/widen_minigrid_hparam.sh > logs/widen_minigrid_hparam.log 2>&1 &
#
# WHY. resume_hparam_sigma.sh took its GPU slot lists from the environment
# when it started (2026-09-20 20:47), so its MiniGrid stage is pinned to the
# four slots that were free then. A MiniGrid ES trial costs about five hours
# against a gymnax trial's eighty minutes, so twelve of them on four slots is
# fifteen hours of the sweep's twenty-eight; on eleven it is under six. The
# slot list cannot be changed inside the running script, so this waits for
# that stage to start and restarts it wider.
#
# It waits for the MiniGrid launcher to appear (the action-reversal stage
# before it still has trials), stops THAT launcher and only the MiniGrid
# trials it started -- by PID, filtered on the run tree, never by a bare
# pattern, because other sessions run the same trainers -- and re-queues the
# stage on GPUS. Restarting costs only what the killed trials had done, which
# is why this fires at the start of the stage rather than later.
#
# GPUS (default "6 6 6 6 4 4"), POLL (default 120 s), DEADLINE (default 18 h).
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPUS_WIDE="${GPUS:-6 6 6 6 4 4}"
POLL="${POLL:-120}"
DEADLINE="${DEADLINE:-64800}"
ROOT=projects/iclr_2027/runs_hparam_minigrid
started=$(date +%s)

mine() {  # PIDs whose PROJECT_ROOT is the MiniGrid sweep: launchers and trials
    for p in $(pgrep -f 'launch.sh|queue_iclr_hparam.sh|minigrid/cli.py' 2>/dev/null); do
        r=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep '^PROJECT_ROOT=' | cut -d= -f2)
        c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
        case "$r$c" in *"$ROOT"*) echo "$p" ;; esac
    done
}

echo "=== $(date -Is) waiting for the MiniGrid stage to start"
while :; do
    if [ -n "$(mine)" ]; then
        echo "=== $(date -Is) MiniGrid stage is up; restarting it on [$GPUS_WIDE]"
        break
    fi
    if [ $(( $(date +%s) - started )) -ge "$DEADLINE" ]; then
        echo "=== $(date -Is) deadline reached and the stage never started; giving up" >&2
        exit 1
    fi
    sleep "$POLL"
done

# Stop the narrow stage: launchers first so nothing is respawned, then its
# trials. A trial writes training_metrics.json only when it finishes, so
# anything killed here is simply re-run by run_condition.
for p in $(mine); do kill "$p" 2>/dev/null || true; done
sleep 10
for p in $(mine); do kill -9 "$p" 2>/dev/null || true; done
sleep 5
echo "=== $(date -Is) narrow stage stopped ($(mine | wc -l) left)"

FAMILY=minigrid NUM_TRIALS=3 GPUS_A="$GPUS_WIDE" \
    bash scripts/train/queue_iclr_hparam.sh
echo "=== $(date -Is) MiniGrid done at width [$GPUS_WIDE]"
grep -hv 'exit=0' logs/launch_hparam_minigrid/status.tsv 2>/dev/null || echo "  no failures"
