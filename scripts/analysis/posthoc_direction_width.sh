#!/usr/bin/env bash
# The basin along the method's own path: child_survival.py direction on the
# 13 ReLU panels of fig:basin (gymnax 2-task noise / physics / actions,
# gymnax 10-task noise, MiniGrid), every arm of the figure, the last five
# probed checkpoints of every trial. PBT on the physics panels is the pbt2
# arm saved under the label pbt, as in the overlap probe.
#
#   GPUS="0 1 2 3 4 5 6 7" scripts/analysis/posthoc_direction_width.sh
#
# -> results/direction_width/raw/<panel>__<arm>.json, logs/ beside it; then
#    .venv/bin/python scripts/analysis/plot_basin_width_direction.py --extract
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
read -ra GPU_LIST <<< "${GPUS:-0}"
LOG=results/direction_width/logs; mkdir -p "$LOG"
ARMS="es ga ppo trac redo cchain pbt"
GYMNAX="cartpole_noise2 acrobot_noise2 mountaincar_noise2 cartpole_actions acrobot_actions mountaincar_actions cartpole_noise10 acrobot_noise10 mountaincar_noise10"
PHYSICS="cartpole_physics2 acrobot_physics2 mountaincar_physics2"
i=0
run() { local g="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"; i=$((i + 1)); echo "gpu $g: $*"; nohup $PY scripts/analysis/child_survival.py direction --phases=-5: --gpus "$g" "$@" > "$LOG/$(echo "$*" | tr ' /' '__').log" 2>&1 & }
# MiniGrid is the slow one (~30 s a checkpoint): one arm a card.
for arm in $ARMS; do run --panels minigrid --arms "$arm"; done
# gymnax: seconds a checkpoint; one job per family of panels, two per card.
run --panels $GYMNAX --arms es ga ppo
run --panels $GYMNAX --arms trac redo cchain pbt
run --panels $PHYSICS --arms es ga ppo trac redo cchain
run --panels $PHYSICS --arms pbt2 --label pbt
wait
echo done
