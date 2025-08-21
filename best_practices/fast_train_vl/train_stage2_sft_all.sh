#! /bin/bash

# 1. 设置环境变量
# WORLD_SIZE=8
# RANK=0

BASE_DATA_PATH="/data/joey.wang/pz_project/xingsen/document_extraction/data/exp/stage3/exp3"
OUTPUT_DIR="./models/stage2_sft_all"

# 2. 运行脚本
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=4 \
MAX_PIXELS=589824 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
swift sft \
    --model ./models/stage1_sft_aligner/v2-20250821-105831/checkpoint-30 \
    --model_type qwen2_5_vl \
    --train_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --eval_steps 5 \
    --save_steps 5 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 4096 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero3 \
    --dataset "$BASE_DATA_PATH/train_class1.json" \
        "$BASE_DATA_PATH/train_class3.json" \
        "$BASE_DATA_PATH/train_class4.json" \
        "$BASE_DATA_PATH/train_class6.json" \
        "$BASE_DATA_PATH/train_class8.json" \
        "$BASE_DATA_PATH/train_class9.json" \
        "$BASE_DATA_PATH/train_class10.json" \
    --val_dataset "$BASE_DATA_PATH/val_class1.json" \
        "$BASE_DATA_PATH/val_class3.json" \
        "$BASE_DATA_PATH/val_class4.json" \
        "$BASE_DATA_PATH/val_class6.json" \
        "$BASE_DATA_PATH/val_class8.json" \
        "$BASE_DATA_PATH/val_class9.json" \
        "$BASE_DATA_PATH/val_class10.json" \