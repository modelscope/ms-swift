#!/bin/bash
# 2 * Ascend 910B3 (64 GiB)
# FSDP2 shards the training model across two data-parallel ranks.

source /usr/local/Ascend/nnal/atb/set_env.sh || exit 1

NPROC_PER_NODE=2 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MASTER_PORT=29501 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-4B \
    --check_model false \
    --dataset 'AI-MO/NuminaMath-TIR#32' \
    --split_dataset_ratio 0 \
    --system 'You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}.' \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_enable_lora true \
    --vllm_gpu_memory_utilization 0.20 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 1024 \
    --max_length 256 \
    --max_completion_length 128 \
    --num_generations 2 \
    --steps_per_generation 1 \
    --tuner_type lora \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing false \
    --learning_rate 1e-5 \
    --fsdp fsdp2 \
    --max_steps 1 \
    --logging_steps 1 \
    --save_strategy no \
    --eval_strategy no \
    --dataloader_num_workers 1 \
    --dataset_num_proc 1 \
    --report_to none \
    --output_dir output/qwen3_5_4b_grpo_fsdp2
