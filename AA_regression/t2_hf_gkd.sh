#!/bin/bash
# Regression smoke test: HF GKD (vLLM colocate)
cd /mnt/nas2/hujinghan.hjh/swift
export PATH=/mnt/nas2/anaconda3/envs/hjh_h20/bin:$PATH
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-0.5B \
    --teacher_model Qwen/Qwen2.5-1.5B-Instruct \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#200' \
    --split_dataset_ratio 0.01 \
    --seq_kd false \
    --lmbda 0.5 \
    --torch_dtype bfloat16 \
    --max_steps 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --max_length 1024 \
    --max_completion_length 256 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --sleep_level 1 \
    --attn_impl flash_attn \
    --output_dir AA_regression/output/t2_hf_gkd \
    --dataloader_num_workers 2 \
    --dataset_num_proc 2
