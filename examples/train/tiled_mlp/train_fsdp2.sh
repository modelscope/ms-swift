#!/bin/bash
# FSDP2 training with tiled MLP
# Requires accelerate config with fsdp_version: 2

# First, create the accelerate config (fsdp2.json) or use the one in examples/train/multi-gpu/fsdp2_lora/

# FSDP2 with tiled MLP
accelerate launch --config_file fsdp2.json \
    -m swift sft \
    --model Qwen/Qwen3-4B \
    --dataset swift/self-cognition#200 \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_checkpointing false \
    --weight_decay 0.1 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --use_tiled_mlp true \
    --tiled_mlp_num_shards 4
