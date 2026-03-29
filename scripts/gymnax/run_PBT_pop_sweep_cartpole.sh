#!/bin/bash
# Run PBT population sweep for CartPole-v1 on GPU 0
ENVS="CartPole-v1" GPU=0 ./scripts/gymnax/run_PBT_gymnax_continual_pop_sweep.sh
