#!/bin/bash
# HF GRPO sequence_parallel_size>1 smoke test.
#
# Validates the SP logps path (_get_logps_via_sp) which reads
# inputs['grpo_batch'].seq_lengths after the collate refactor.
#
# Minimal: Qwen2-0.5B + LoRA + 2 steps + SP=2 on 2 GPUs (4,5).
set -x
export CUDA_VISIBLE_DEVICES=4,5
export NPROC_PER_NODE=2
export PYTORCH_CUDA_ALLOC_CONF=''

cd /mnt/nas2/hujinghan.hjh/swift

/mnt/nas2/anaconda3/envs/hjh_h20/bin/swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2-0.5B \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#100' \
    --reward_funcs format \
    --num_generations 2 \
    --max_completion_length 128 \
    --max_length 1024 \
    --max_steps 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_steps 2 \
    --split_dataset_ratio 0.01 \
    --logging_steps 1 \
    --use_vllm false \
    --eval_strategy no \
    --padding_free true \
    --attn_impl flash_attn \
    --sequence_parallel_size 2 \
    --dataloader_drop_last true \
    --output_dir output/_smoke_sp
