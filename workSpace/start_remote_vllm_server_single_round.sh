#!/bin/bash

# 远程vLLM服务器启动脚本
# 在远程rollout机器上运行此脚本

# 模型配置
MODEL_NAME="/mnt/cfs/ssw/ljc/LLaMA-Factory/saves/qwen3-4b/full/4B_1_0_1"  # 基线模型地址
HOST="0.0.0.0"  # 监听所有网络接口
PORT=8000
GPU_MEMORY_UTILIZATION=0.95
# TENSOR_PARALLEL_SIZE=8
TENSOR_PARALLEL_SIZE=2

# 设置CUDA设备
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export CUDA_VISIBLE_DEVICES=0,1

export NCCL_DEBUG=WARN
# ROLLOUT 日志设置（可通过环境变量覆盖）
export ROLLOUT_LOG_LEVEL=${ROLLOUT_LOG_LEVEL:-INFO}
export ROLLOUT_LOG_PATH=${ROLLOUT_LOG_PATH:-/mnt/cfs/ssw/ljc/ms-swift/workSpace/GRPO/log/test.log}
# # 创建目录；若无权限则回退到 ~/.ms-swift/rollout.log
# mkdir -p "$(dirname "$ROLLOUT_LOG_PATH")" 2>/dev/null || true
# if ! touch "$ROLLOUT_LOG_PATH" 2>/dev/null; then
#     mkdir -p "$HOME/.ms-swift" 2>/dev/null || true
#     export ROLLOUT_LOG_PATH="$HOME/.ms-swift/rollout.log"
#     touch "$ROLLOUT_LOG_PATH" 2>/dev/null || echo "无法创建日志文件：$ROLLOUT_LOG_PATH"
# fi

echo "正在启动vLLM Rollout服务器..."
echo "模型: $MODEL_NAME (基线权重，训练中会动态更新)"
echo "监听地址: $HOST:$PORT"
echo "GPU内存利用率: $GPU_MEMORY_UTILIZATION"
echo "张量并行度: $TENSOR_PARALLEL_SIZE"
echo "日志: $ROLLOUT_LOG_PATH (level=$ROLLOUT_LOG_LEVEL)"

export LOG_LEVEL=DEBUG
export SWIFT_DEBUG=true

# 启动vLLM服务器（禁用多轮对话，仅单轮工具调用）
swift rollout \
    --model $MODEL_NAME \
    --model_type 'qwen3' \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --enable-prefix-caching true \
    --enforce-eager false \
    --enable-prefix-caching true \
    --served-model-name $MODEL_NAME \
    --max-num-seqs 256 \
    --max-model-len 32768 \
    --max-turns 1 \
    --use-async-engine false 

# 移除的参数（ms-swift 不支持）：
# --trust-remote-code true
# --disable-log-stats $DISABLE_LOG_STATS
# --enable-chunked-prefill $ENABLE_CHUNKED_PREFILL
# --disable-sliding-window
# --swap-space 6
# --block-size 16
# --rope-scaling (格式可能不兼容) 
# --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \ 
# --multi-turn-scheduler null \

# use_async_engine: vLLM backend下是否使用async engine。
    # 部署情况（swift deploy）默认为True，其他情况默认为False。