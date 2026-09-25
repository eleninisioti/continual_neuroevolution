#!/bin/bash
# ============================================================================
# Tune NES (the paper's "ES" on gymnax) on STATIONARY MountainCar.
#
# WHY. The reported NES setting (sigma 0.1, lr 0.05: NES_ENV_CONFIGS in
# source/studies/gymnax/es_algorithms.py, inherited from the generalists
# study) does not solve stationary MountainCar: over the 10 reported trials
# (runs_centroid/gymnax/noncontinual/nes) the centroid peaks at -114..-156 and
# ends at -173..-214, never at the -100 that GA, OpenES and PPO all reach. In
# the first, unperturbed 200-generation phase of the continual runs it finds
# the goal (peaks -124..-196) and has lost it again by the switch (7 of 10
# trials end the phase at -500). The Figure 2 MountainCar panel therefore
# compares a mis-tuned ES against tuned everything-else. The existing width
# sweep (runs_hparam nes_sigma*, lr fixed at 0.05) does not fix it: x0.1/x0.3
# never learn, x3 is too slow to learn inside a phase, x10 collapses.
#
# The failure looks like step size, not width: with standardised fitness the
# centroid moves ~lr/sigma a generation, and one -500 timeout among 512
# solvers gets a utility of about -20 and kicks the centroid along a random
# direction. So this sweeps width and step TOGETHER, over the ordinary
# settings only (no sigma adaptation, no new terms), on the same budget as
# every reported stationary run: 600 generations x 512 x 3 evals, 16x16 relu,
# 10 report episodes, seeds 42.. (trials 1-5 = the reported trials' seeds).
#
#     sigma   0.05    lr 0.005 0.01 0.02 0.05
#     sigma   0.1     lr 0.005 0.01 0.02 0.1      (0.05 = the reported run)
#     sigma   0.2     lr 0.01  0.02 0.05 0.1
#     sigma   0.3     lr 0.02  0.05 0.1  0.15
#
# Tree: projects/iclr_2027/runs_hparam/nes_mcar_tune/gymnax/noncontinual/
#       nes_sigma<S>_lr<L>/MountainCar_v0/trial_k   (value in the arm name,
#       run_experiments.sh block_gymnax_noncontinual "HYPERPARAMETER ARMS")
# Read: scripts/analysis/nes_mcar_tune.py  (table + curves)
#
# A trial is ~7 min alone, more on a shared card. 16 arms x 5 trials = 80
# jobs; at 12 slots about 1-2 h. Env: GPUS (slots; repeat an index to run
# several jobs on one card), NUM_TRIALS (5), ARMS, DRY_RUN=1.
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT=projects/iclr_2027/runs_hparam/nes_mcar_tune
NUM_TRIALS="${NUM_TRIALS:-5}"
export DRY_RUN="${DRY_RUN:-0}"
export TRACK_DIVERSITY=1

ARMS="${ARMS:-\
nes_sigma0.05_lr0.005 nes_sigma0.05_lr0.01 nes_sigma0.05_lr0.02 nes_sigma0.05_lr0.05 \
nes_sigma0.1_lr0.005 nes_sigma0.1_lr0.01 nes_sigma0.1_lr0.02 nes_sigma0.1_lr0.1 \
nes_sigma0.2_lr0.01 nes_sigma0.2_lr0.02 nes_sigma0.2_lr0.05 nes_sigma0.2_lr0.1 \
nes_sigma0.3_lr0.02 nes_sigma0.3_lr0.05 nes_sigma0.3_lr0.1 nes_sigma0.3_lr0.15}"

echo "=== $(date -Is) NES MountainCar tuning sweep -> $ROOT ==="
NUM_TRIALS="$NUM_TRIALS" PROJECT_ROOT="$ROOT" ENVS=MountainCar-v0 \
  GPUS="${GPUS:-5 5 5 5 6 6 6 6 7 7 7 7}" \
  LOG_DIR="logs/launch_nes_mcar_tune" \
  bash scripts/train/launch.sh gymnax_noncontinual $ARMS
echo "=== $(date -Is) done ==="
grep -hv 'exit=0' logs/launch_nes_mcar_tune/status.tsv 2>/dev/null || echo "  no failures"
