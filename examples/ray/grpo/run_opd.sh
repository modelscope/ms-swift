#!/bin/bash
# Ray Megatron OPD-RL (On-Policy Distillation as RL) — colocate mode + colocated teacher.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

megatron rlhf --use_ray true --config "$SCRIPT_DIR/opd_rl_colocate.yaml"
