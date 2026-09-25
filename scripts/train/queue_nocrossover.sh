#!/bin/bash
# Re-run the paper's ga_focus GA without crossover (`ga_focus_explore_nox`,
# source/studies/kinetix/settings.py): the probe (2026-09-21, one seed each)
# showed crossover does not help on MountainCar or Kinetix, so the paper drops it.
#
#   MountainCar  every family in runs_ga_focus_mountaincar, 10 trials, into the
#                same tree under the ga_focus_explore_nox arm. Small jobs,
#                shared cards (MC_GPUS, MC_PER_GPU).
#   Kinetix      the continual chain (trials 1-4) and the twenty stationary
#                levels (trials 1-5), seeds 41 + trial as in the reported runs,
#                into runs_nocrossover. One job per card, started only when a
#                card has no compute process at all (the chain takes ~13 h on
#                an idle card).
#
# Finished runs are skipped; a continual chain resumes from its resume.pkl.
# Stop Kinetix chains with the STOP file, never a kill (the chain is lost).
#
#   nohup bash scripts/train/queue_nocrossover.sh > logs/nocrossover.log 2>&1 &
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=.venv/bin/python
ARM=ga_focus_explore_nox
KX_ROOT=projects/iclr_2027/runs_nocrossover/kinetix
LOGS=projects/iclr_2027/runs_nocrossover/logs
MC_GPUS=(${MC_GPUS:-7 6})
MC_PER_GPU=${MC_PER_GPU:-3}
KX_GPUS="${KX_GPUS:-0 1 2 3 4 5 6 7}"
mkdir -p "$LOGS"

# --- MountainCar ------------------------------------------------------------
(
i=0
for family in noise_10task actions_2task noise_2task physics_10task physics_2task stationary; do
    for trial in 1 2 3 4 5 6 7 8 9 10; do
        echo "$family $trial ${MC_GPUS[$(( i % ${#MC_GPUS[@]} ))]}"
        i=$(( i + 1 ))
    done
done | ARM=$ARM xargs -P $(( ${#MC_GPUS[@]} * MC_PER_GPU )) -L 1 bash -c \
    'nice -n 5 '"$PY"' scripts/train/ga_focus_mountaincar.py "$0" "$1" "$2" \
        > '"$LOGS"'/mc_"$0"_t"$1".log 2>&1 && echo "done mc $0 trial $1" || echo "FAILED mc $0 trial $1"'
echo "=== $(date -Is) MountainCar finished"
) &

# --- Kinetix ----------------------------------------------------------------
LEVELS=$($PY -c "from source.studies.kinetix import settings as S; print(' '.join(S.NONCONTINUAL_CELLS))")
JOBS=()
for t in 1 2 3 4; do JOBS+=("Kinetix20 $t"); done
for t in 1 2 3 4 5; do for lv in $LEVELS; do JOBS+=("$lv $t"); done; done

idle_gpu() {  # first card in KX_GPUS with no compute process and not ours-in-flight
    local busy
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    while IFS=, read -r idx uuid; do
        uuid=${uuid// /}
        [[ " $KX_GPUS " == *" $idx "* ]] || continue
        grep -q "$uuid" <<<"$busy" && continue
        [ -e "$LOGS/.claim_gpu$idx" ] && continue
        echo "$idx"; return 0
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    return 1
}

for job in "${JOBS[@]}"; do
    set -- $job; env=$1; t=$2
    if [ "$env" = Kinetix20 ]; then
        out=$KX_ROOT/continual/$ARM/Kinetix20/trial_$t; extra="--checkpoint_every 200"
    else
        out=$KX_ROOT/noncontinual/$ARM/${env/-/_}/trial_$t; extra=""
    fi
    [ -f "$out/training_metrics.json" ] && { echo "skip $out"; continue; }
    until g=$(idle_gpu); do sleep 60; done
    mkdir -p "$out"; touch "$LOGS/.claim_gpu$g"
    echo "=== $(date -Is) GPU $g: $env trial $t"
    ( CUDA_VISIBLE_DEVICES=$g XLA_PYTHON_CLIENT_PREALLOCATE=false nice -n 5 \
        $PY source/studies/kinetix/cli.py --env "$env" --method $ARM \
        --trial "$t" --seed $(( 41 + t )) --gpus "$g" $extra --output_dir "$out" \
        > "$LOGS/kx_${env}_t$t.log" 2>&1 \
        && echo "done kx $env trial $t" || echo "FAILED kx $env trial $t"
      rm -f "$LOGS/.claim_gpu$g" ) &
    sleep 90   # let it register on the card before the next idle check
done
wait
echo "=== $(date -Is) queue finished"
