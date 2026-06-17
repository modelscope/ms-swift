#!/bin/bash
# Regression smoke test: HF GRPO (vLLM colocate) + off-policy metrics
cd /mnt/nas2/hujinghan.hjh/swift
export PATH=/mnt/nas2/anaconda3/envs/hjh_h20/bin:$PATH
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --reward_funcs accuracy \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#100' \
    --load_from_cache_file true \
    --max_completion_length 512 \
    --max_length 1024 \
    --max_steps 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --num_generations 4 \
    --temperature 1.0 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_max_model_len 1024 \
    --log_rollout_offpolicy_metrics true \
    --log_completions true \
    --offload_model true \
    --offload_optimizer true \
    --sleep_level 1 \
    --output_dir AA_regression/output/t1_hf_grpo \
    --dataloader_num_workers 2 \
    --dataset_num_proc 2
