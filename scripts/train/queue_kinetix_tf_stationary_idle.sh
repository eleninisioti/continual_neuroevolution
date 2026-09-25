#!/bin/bash
# The transformer Kinetix STATIONARY block on whatever home cards are free,
# while the Kinetix20 chains still hold the rest. Same runs as
# queue_iclr_kinetix_transformer.sh STAGES=ne_noncont / rl_noncont (same
# layout, seeds 41 + trial, --observation entity), but it takes a card only
# when no chain is on it, and puts at most PER_CARD stationary jobs there.
# Cards in RESERVE are skipped while any process matching RESERVE_WHILE runs
# (the ES chains being moved onto 6 and 7).
#
#   ARMS="ga_focus_explore_nox es_hold" nohup bash \
#       scripts/train/queue_kinetix_tf_stationary_idle.sh > logs/kx_tf_stat.log 2>&1 &
#
# A job whose training_metrics.json exists is skipped, so re-running resumes.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="${ROOT:-projects/iclr_2027/runs_kinetix_tf}/kinetix/noncontinual"
ARMS="${ARMS:-ga_focus_explore_nox es_hold}"
TRIALS="${TRIALS:-1 2 3}"
PER_CARD="${PER_CARD:-2}"
RESERVE="${RESERVE:-6 7}"
RESERVE_WHILE="${RESERVE_WHILE:-move_es.sh}"
PY=.venv/bin/python
LOGS=logs/kinetix_tf_stationary
mkdir -p "$LOGS"
LEVELS=$($PY -c "from source.envs.kinetix_levels import LEVELS; print(' '.join(LEVELS))")

free_card() {   # a card with no chain and < PER_CARD stationary jobs
    local g n
    for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
        if [[ " $RESERVE " == *" $g "* ]] && pgrep -f "$RESERVE_WHILE" > /dev/null; then
            continue
        fi
        pgrep -f "cli.py --env Kinetix20 --gpus $g " > /dev/null && continue
        n=$(pgrep -fc "cli.py --env Kinetix-[^ ]* --gpus $g ")
        [ "$n" -lt "$PER_CARD" ] && [ ! -e "$LOGS/.claim$g" ] && { echo "$g"; return 0; }
    done
    return 1
}

for arm in $ARMS; do for lv in $LEVELS; do for t in $TRIALS; do
    out=$ROOT/$arm/Kinetix_$lv/trial_$t
    [ -f "$out/training_metrics.json" ] && continue
    until g=$(free_card); do sleep 30; done
    touch "$LOGS/.claim$g"
    mkdir -p "$out"
    echo "$(date -Is) GPU $g: $arm $lv trial $t"
    ( XLA_PYTHON_CLIENT_PREALLOCATE=false $PY source/studies/kinetix/cli.py \
        --env "Kinetix-$lv" --gpus "$g" --trial "$t" --seed $((41 + t)) \
        --method "$arm" --observation entity --output_dir "$out" \
        > "$LOGS/${arm}_${lv}_t$t.log" 2>&1 \
        && echo "$(date -Is) done $arm $lv t$t" \
        || echo "$(date -Is) FAILED $arm $lv t$t" ) &
    sleep 20; rm -f "$LOGS/.claim$g"   # registered on the card by now
done; done; done
wait
echo "$(date -Is) all stationary jobs finished"
