#!/bin/bash
# 2 * Ascend 910B3 (64 GiB)
# TP=PP=1, so both ranks are data-parallel training ranks.
# generation_batch_size = global_batch_size * steps_per_generation = 4.
# rollout prompts = generation_batch_size / num_generations = 2.

source /usr/local/Ascend/nnal/atb/set_env.sh || exit 1

NPROC_PER_NODE=2 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MASTER_PORT=29502 \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-4B \
    --check_model false \
    --save_safetensors true \
    --dataset 'AI-MO/NuminaMath-TIR#32' \
    --split_dataset_ratio 0 \
    --system 'You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}.' \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.20 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 1024 \
    --max_length 256 \
    --max_completion_length 128 \
    --num_generations 2 \
    --steps_per_generation 1 \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --global_batch_size 4 \
    --micro_batch_size 1 \
    --tuner_type lora \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 32 \
    --merge_lora true \
    --bf16 true \
    --gradient_accumulation_fusion false \
    --lr 1e-5 \
    --finetune true \
    --train_iters 1 \
    --logging_steps 1 \
    --save_steps 1000 \
    --no_save_optim \
    --no_save_rng \
    --dataloader_num_workers 1 \
    --dataset_num_proc 1 \
    --attention_backend flash \
    --padding_free false \
    --log_completions true \
    --output_dir output/qwen3_5_4b_grpo_megatron
