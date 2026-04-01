#!/bin/bash
# =============================================================================
# Geometry3K VLM GRPO Training with Qwen3.5-2B
# =============================================================================
# Dataset: hiyouga/geometry3k (built-in registered in swift)
#   - Fields: problem -> query, answer -> solution, images
#   - Train: 2101, Val: 300, Test: 601
#   - Task: Visual geometry math reasoning (given a geometry diagram, answer the question)
#
# Reward Functions (built-in):
#   - accuracy (MathAccuracy): math_verify + boxed/answer_tag extraction -> 0/1
#   - format (Format): checks <think>...</think><answer>...</answer> -> 0/1
#
# Model: Qwen/Qwen3.5-2B (native multimodal, hybrid thinking)
#
# Mode: Single GPU, LoRA, vLLM colocate (sleep_level=1)
#
# Prerequisites:
#   pip install math_verify latex2sympy2_extended
# =============================================================================

SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=401408 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-2B \
    --reward_funcs accuracy format \
    --reward_weights 1.0 0.1 \
    --enable_thinking false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.45 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --sleep_level 1 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset 'hf::hiyouga/geometry3k:train' \
    --val_dataset 'hf::hiyouga/geometry3k:test' \
    --load_from_cache_file true \
    --max_length 2048 \
    --max_completion_length 1024 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --save_steps 50 \
    --save_total_limit 5 \
    --eval_strategy steps \
    --eval_steps 50 \
    --logging_steps 1 \
    --output_dir output/grpo_geo3k_qwen35_2b \
    --dataloader_num_workers 4 \
    --num_generations 4 \
    --temperature 1.0 \
    --system "$SYSTEM_PROMPT" \
    --log_completions true \
    --report_to tensorboard \
    --max_grad_norm 1.0 \
    --beta 0.001
