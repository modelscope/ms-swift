#!/bin/bash
# Regression smoke test: Megatron GRPO (vLLM colocate) + off-policy metrics
cd /mnt/nas2/hujinghan.hjh/swift
export PATH=/mnt/nas2/anaconda3/envs/hjh_h20/bin:$PATH
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
MASTER_PORT=29600 \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --dataset 'AI-MO/NuminaMath-TIR#100' \
    --reward_funcs accuracy \
    --global_batch_size 16 \
    --micro_batch_size 2 \
    --steps_per_generation 1 \
    --num_generations 4 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 2048 \
    --max_length 1024 \
    --max_completion_length 512 \
    --tuner_type full \
    --lr 1e-6 \
    --bf16 true \
    --beta 0.001 \
    --importance_sampling_level token \
    --epsilon 0.2 \
    --epsilon_high 0.2 \
    --loss_type grpo \
    --log_rollout_offpolicy_metrics true \
    --sleep_level 2 \
    --offload_model true \
    --offload_optimizer true \
    --logging_steps 1 \
    --recompute_granularity selective \
    --finetune \
    --no_save_optim \
    --no_save_rng \
    --attention_backend flash \
    --temperature 1.0 \
    --padding_free true \
    --log_completions true \
    --train_iters 5 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --output_dir megatron_output/t3_mg_grpo \
    --dataloader_num_workers 2 \
    --dataset_num_proc 2
