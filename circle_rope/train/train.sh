#!/bin/bash

# Circle-RoPE Training Script (DDP mode)
nproc_per_node=4

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --train_config circle_rope/exp/train_circle.json
