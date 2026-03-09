#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_DIR="/root/autodl-tmp/model/Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_DIR="/root/autodl-tmp/sft_data/train/grpo_result"
DATASET="/root/autodl-tmp/sft_data/train/dataset_multi_vio.json"


NPROC_PER_NODE=4
TOTAL_BATCH_SIZE=16
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / (NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE)))

FORCE_TORCHRUN=1 \
NPROC_PER_NODE=$NPROC_PER_NODE \
swift sft \
    --model $MODEL_DIR \
    --model_type qwen3_vl \
    --train_type lora \
    --dataset $DATASET \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate 1e-4 \
    --save_steps 30 \
    --save_total_limit 50 \
    --logging_steps 10 \
    --max_length 8192 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --bf16 true \
    --save_safetensors true \
    --gradient_checkpointing true \
