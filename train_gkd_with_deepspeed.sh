#!/bin/bash
# 使用 DeepSpeed 启动训练脚本

export WANDB_API_KEY=28e11ef52849c4640b93051377be27eafac62c44
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0

# 使用 torchrun 启动以支持 DeepSpeed
# 注意: 在运行前,需要在 train_gkd_debug.py 中启用 DeepSpeed 配置
torchrun --nproc_per_node=1 train_gkd_debug.py
