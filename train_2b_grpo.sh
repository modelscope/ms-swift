#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model /root/autodl-tmp/model/Qwen3_grpo \
    --dynamic_sample true \
    --max_resample_times 3 \
    --train_type lora \
    --dataset /root/autodl-tmp/sft_data/train/dataset_rlhf_new.json \
    --use_vllm true \
    --vllm_mode server \
    --vllm_gpu_memory_utilization 0.55 \
    --vllm_tensor_parallel_size 2 \
    --vllm_mm_processor_cache_gb 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-6 \
    --save_total_limit 20 \
    --logging_steps 5 \
    --output_dir /root/autodl-tmp/sft_data/train/grpo_result \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_completion_length 1024 \
    --external_plugins /root/autodl-tmp/sft_data/train/safety_reward_plugin1.py \
    --reward_funcs multi_label_safety_penalty format \
    --num_generations 8 \
    --sleep_level 0 \
    --temperature 0.7 \
    --top_p 0.85 \
    --deepspeed zero3