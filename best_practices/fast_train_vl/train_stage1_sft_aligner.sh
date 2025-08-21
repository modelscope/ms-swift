#! /bin/bash

# 1. 设置环境变量
# WORLD_SIZE=1
# RANK=0

BASE_DATA_PATH="/data/joey.wang/pz_project/xingsen/document_extraction/data/exp/stage3/exp3"
OUTPUT_DIR="./models/stage1_sft_aligner"

# 2. 运行脚本
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=4 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
swift sft \
    --model ./Qwen3-VL-Model \
    --model_type qwen2_5_vl \
    --train_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --freeze_vit true \
    --freeze_llm true \
    --freeze_aligner false \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --eval_steps 5 \
    --save_steps 5 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 8192 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero2 \
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