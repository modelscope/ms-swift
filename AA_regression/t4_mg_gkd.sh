#!/bin/bash
# Regression smoke test: Megatron GKD (dense)
cd /mnt/nas2/hujinghan.hjh/swift
export PATH=/mnt/nas2/anaconda3/envs/hjh_h20/bin:$PATH
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
MASTER_PORT=29610 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3-0.6B \
    --teacher_model Qwen/Qwen3-1.7B \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#200' \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --micro_batch_size 2 \
    --global_batch_size 8 \
    --lr 5e-6 \
    --logging_steps 1 \
    --max_length 2048 \
    --max_completion_length 1024 \
    --attention_backend flash \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --sleep_level 1 \
    --offload_teacher_model true \
    --recompute_granularity selective \
    --finetune \
    --no_save_optim \
    --no_save_rng \
    --temperature 1.0 \
    --padding_free true \
    --train_iters 5 \
    --output_dir megatron_output/t4_mg_gkd \
    --dataset_num_proc 2
