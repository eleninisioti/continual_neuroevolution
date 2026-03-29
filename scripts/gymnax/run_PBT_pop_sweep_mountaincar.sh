#!/bin/bash
# Run PBT population sweep for MountainCar-v0 on GPU 2
ENVS="MountainCar-v0" GPU=2 ./scripts/gymnax/run_PBT_gymnax_continual_pop_sweep.sh
