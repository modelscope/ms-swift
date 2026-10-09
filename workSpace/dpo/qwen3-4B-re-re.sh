#!/bin/bash

# # 设置缓存和临时文件目录到有更多空间的位置
# export TMPDIR=/mnt/cfs/ssw/ljc/tmp
# export HF_HOME=/mnt/cfs/ssw/ljc/cache/huggingface
# export TRANSFORMERS_CACHE=/mnt/cfs/ssw/ljc/cache/transformers
# export HF_DATASETS_CACHE=/mnt/cfs/ssw/ljc/cache/datasets
# export TORCH_HOME=/mnt/cfs/ssw/ljc/cache/torch

# # 创建缓存目录（如果不存在）
# mkdir -p $TMPDIR
# mkdir -p $HF_HOME
# mkdir -p $TRANSFORMERS_CACHE
# mkdir -p $HF_DATASETS_CACHE
# mkdir -p $TORCH_HOME

NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type dpo \
    --model /mnt/cfs/ssw/ljc/ms-swift/output/v8-20250806-224940/checkpoint-44 \
    --model_type qwen3 \
    --train_type full \
    --dataset /mnt/cfs/ssw/ljc/dataset_making/Data/ready_dataset/RLHF/0807/msswift_dpo_fixed.json \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps 5 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 16000 \
    --output_dir output \
    --warmup_ratio 0.1 \
    --save_only_model true \
    --dataloader_num_workers 0 \
    --dataset_num_proc 8 \
    --attn_impl flash_attn \
    --agent_template hermes \
    --deepspeed zero2 \
    --report_to swanlab \
    --swanlab_token GFPjNmyR2K5Cog3C6N7uA \
    --swanlab_mode cloud \
    --strict true 
