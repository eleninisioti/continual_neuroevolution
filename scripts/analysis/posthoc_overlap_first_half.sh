#!/usr/bin/env bash
# The shared-basin probe (child_survival.py overlap) on the FIRST half of every
# run (checkpoints 0..8), the half the paper's probe skipped, so that
# plot_shared_basin_evolution.py can draw the shared basin over training. Same
# 13 ReLU panels and arms as posthoc_direction_width.sh. Written to its own
# folder so the paper's second-half figures (results/overlap_probe) do not change.
#
#   GPUS="0 1 2 3 4 5 6 7" scripts/analysis/posthoc_overlap_first_half.sh
#
# -> results/overlap_first_half/raw/<panel>__<arm>.json, logs/ beside it
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
read -ra GPU_LIST <<< "${GPUS:-0}"
LOG=results/overlap_first_half/logs; mkdir -p "$LOG"
ARMS="es ga ppo trac redo cchain pbt"
GYMNAX="cartpole_noise2 acrobot_noise2 mountaincar_noise2 cartpole_actions acrobot_actions mountaincar_actions cartpole_noise10 acrobot_noise10 mountaincar_noise10"
PHYSICS="cartpole_physics2 acrobot_physics2 mountaincar_physics2"
i=0
run() { local g="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"; i=$((i + 1)); echo "gpu $g: $*"; nohup $PY scripts/analysis/child_survival.py overlap --phases :9 --out overlap_first_half --gpus "$g" "$@" > "$LOG/$(echo "$*" | tr ' /' '__').log" 2>&1 & }
for arm in $ARMS; do run --panels minigrid --arms "$arm"; done
run --panels $GYMNAX --arms es ga ppo
run --panels $GYMNAX --arms trac redo cchain pbt
run --panels $PHYSICS --arms es ga ppo trac redo cchain
run --panels $PHYSICS --arms pbt2 --label pbt
wait
echo done
