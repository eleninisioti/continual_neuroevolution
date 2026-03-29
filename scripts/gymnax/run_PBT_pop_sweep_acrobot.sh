#!/bin/bash
# Run PBT population sweep for Acrobot-v1 on GPU 1
ENVS="Acrobot-v1" GPU=1 ./scripts/gymnax/run_PBT_gymnax_continual_pop_sweep.sh
