#!/bin/bash

echo "拉起 external vLLM server..."

CUDA_VISIBLE_DEVICES=6 \
swift rollout \
    --model /data/joey.wang/llm/models/pretrained/llm/Qwen2.5-VL-7B-Instruct \
    # --data_parallel_size 2
