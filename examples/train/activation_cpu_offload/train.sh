#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --config_file "./examples/train/activation_cpu_offload/fsdp2.json" \
    swift/cli/sft.py \
    --model 'Qwen/Qwen3-0.6B' \
    --train_type lora \
    --dataset 'swift/self-cognition#1000' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing false \ // no need to checkpoint activations when offloading to CPU
    --max_length 1200 \
    --num_train_epochs 2 \
    --eval_strategy no \
    --save_steps 500 \
    --logging_steps 1 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output \
    --attn_impl 'flash_attention_2' \
    --packing true \
    --activation_cpu_offload true
