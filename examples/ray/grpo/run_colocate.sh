#!/bin/bash
# Ray-based Megatron GRPO — colocate mode
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

megatron rlhf --use_ray true --config "$SCRIPT_DIR/ray_grpo_colocate.yaml"
