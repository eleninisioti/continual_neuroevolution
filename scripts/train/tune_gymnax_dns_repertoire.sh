#!/bin/bash
# ============================================================================
# Pilot: is GA + Novelty (`dns_gaussian`) on gymnax held back by its untuned
# repertoire size? At --repertoire_ratio 0.5 half the 512 evaluations re-score
# a repertoire whose mean fitness sits near the floor (CartPole 215 vs the
# GA's 479), and parents are drawn uniformly from it. A setting, not an
# algorithm change: 0.25 -> 128 kept + 384 offspring, 0.1 -> 51 + 461.
#
# Cells: the three where diversity_metrics_elite does not show a gain
# (Acrobot noise, MountainCar noise 0.1, CartPole actions) plus Acrobot
# actions, the largest win, to check the smaller repertoire keeps it.
# Everything else as runs_{centroid,actions}_dnsrefresh (20 sub-tasks,
# period 10, 200 generations each, diversity tracked).
#
#   -> projects/iclr_2027/tune_dns_repertoire/r<ratio>_<family>/gymnax/continual/dns_gaussian/
#
#   GPUS_A / GPUS_B / GPUS_C: one GPU index each (default 0 1 2)
#   bash scripts/train/tune_gymnax_dns_repertoire.sh
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
R=projects/iclr_2027/tune_dns_repertoire
A="${GPUS_A:-0}"; B="${GPUS_B:-1}"; C="${GPUS_C:-2}"
export NUM_TRIALS="${NUM_TRIALS:-3}" TRACK_DIVERSITY=1
export NUM_TASKS=20 TASK_PERIOD=10 TASK_INTERVAL=200

go() {  # ratio family envs sigma gpus
    DNS_REPERTOIRE_RATIO=$1 PROJECT_ROOT=$R/r$1_$2 ENVS="$3" SIGMAS=$4 GPUS="$5" \
    TASK_TYPE=$2 LOG_DIR=logs/tune_dns_repertoire/r$1_$2_${3%%-*} \
        bash scripts/train/launch.sh gymnax_continual dns_gaussian &
}
# launch.sh leases are per launcher: every launcher gets its own slots.
go 0.25 noise   Acrobot-v1              1.0 "$A $A"
go 0.25 noise   MountainCar-v0          0.1 "$A $B"
go 0.25 actions "CartPole-v1 Acrobot-v1" 1.0 "$A $B $C"
go 0.1  noise   Acrobot-v1              1.0 "$B $B"
go 0.1  noise   MountainCar-v0          0.1 "$C $C"
go 0.1  actions "CartPole-v1 Acrobot-v1" 1.0 "$A $B $C"
wait
echo "done $(date -Is)"
grep -hv 'exit=0' logs/tune_dns_repertoire/*/status.tsv 2>/dev/null || echo "no failures"
