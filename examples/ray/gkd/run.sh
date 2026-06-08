#!/bin/bash
# Ray Megatron GKD — default example (rollout colocate + colocated teacher).
# Swap --config for another yaml in this folder for other placements/teacher modes.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

megatron rlhf --use_ray true --config "$SCRIPT_DIR/rollout_colocate_teacher_colocate.yaml"
