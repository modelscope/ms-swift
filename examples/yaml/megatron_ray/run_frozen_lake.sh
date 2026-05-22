#!/bin/bash
# [Experimental] Ray-based Megatron GRPO — FrozenLake multi-turn (colocate mode).
#
# Mirrors the plain-Megatron script at
# examples/megatron/grpo/multi_turn/frozen_lake.sh, but routes training through
# the Ray driver (swift/ray/megatron/grpo_trainer.py) which dispatches rollout
# to RolloutReplica actors.  The agent loop (run_multi_turn) runs on the driver
# side; env code lives in examples/megatron/grpo/multi_turn/frozen_lake_plugin.py.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

megatron rlhf --use_ray true --config "$SCRIPT_DIR/ray_grpo_frozen_lake.yaml"
