#!/bin/bash

# 远程vLLM服务器启动脚本（修复版本）
# 在远程rollout机器上运行此脚本

# 模型配置
MODEL_NAME="/mnt/cfs/ssw/ljc/LLaMA-Factory/saves/qwen3-4b/full/4B_1_0_1"  # 基线模型地址
HOST="0.0.0.0"  # 监听所有网络接口
PORT=8000
GPU_MEMORY_UTILIZATION=0.95
TENSOR_PARALLEL_SIZE=8

# ⭐ 关键配置：Agent Template（必须与训练脚本保持一致）
AGENT_TEMPLATE="qwen_en"  # 或者 qwen_zh（如果是中文数据）

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 根据远程机器GPU配置调整
export NCCL_DEBUG=WARN

# ROLLOUT 日志设置（可通过环境变量覆盖）
export ROLLOUT_LOG_LEVEL=${ROLLOUT_LOG_LEVEL:-INFO}
export ROLLOUT_LOG_PATH=${ROLLOUT_LOG_PATH:-/mnt/cfs/ssw/ljc/ms-swift/workSpace/GRPO/log/test.log}

echo "正在启动vLLM Rollout服务器..."
echo "模型: $MODEL_NAME (基线权重，训练中会动态更新)"
echo "监听地址: $HOST:$PORT"
echo "GPU内存利用率: $GPU_MEMORY_UTILIZATION"
echo "张量并行度: $TENSOR_PARALLEL_SIZE"
echo "Agent Template: $AGENT_TEMPLATE"
echo "日志: $ROLLOUT_LOG_PATH (level=$ROLLOUT_LOG_LEVEL)"

# 启动vLLM服务器（添加agent_template参数）
swift rollout \
    --model $MODEL_NAME \
    --model_type 'qwen3' \
    --agent_template $AGENT_TEMPLATE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --enable-prefix-caching true \
    --use-async-engine true \
    --enforce-eager false \
    --served-model-name $MODEL_NAME \
    --max-num-seqs 256 \
    --max-model-len 32768

# 说明：
# 1. --agent_template 参数确保rollout服务器使用正确的工具调用格式
# 2. 必须与训练脚本中的 --agent_template 参数保持一致
# 3. 这样训练和推理才能使用相同的工具调用格式 