#!/bin/bash
# FSDP2 LoRA Training with CPU Offload
# Similar to DeepSpeed zero3_offload, but uses PyTorch native FSDP2
# Memory: ~10GiB * 2 GPUs (with CPU offload)
# Requires: torch>=2.4.0

nproc_per_node=2

# FSDP2 with CPU offload (similar to --deepspeed zero3_offload)
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --fsdp fsdp2_offload \
    --dataset 'swift/self-cognition#1000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_checkpointing true \
    --weight_decay 0.1 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
