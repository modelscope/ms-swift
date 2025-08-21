#!/bin/bash

echo "开始使用微调后的模型进行流式推理..."

CKPT_DIR="./models/stage2_sft_all"

CUDA_VISIBLE_DEVICES=0 swift infer \
     --model ${CKPT_DIR} \
     --stream True \
     --infer_backend vllm \
     --limit_mm_per_prompt '{"image": 3, "video": 1}' \
     --gpu_memory_utilization 0.8 \
     --max_model_len 128000 \
     --max_new_tokens 4096 \
     --temperature 0 \
     --seed 42 \
