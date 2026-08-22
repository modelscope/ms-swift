#!/bin/bash
# 10.217.16.160   wangxiang4-bce-a800-1-0 训练机
# 10.217.16.121   wangxiang4-bce-a800-2-0 推理机

# hard settings
nproc_per_node=8  # 使用的GPU数量，根据你的硬件调整
model_name="/mnt/cfs/ssw/ljc/LLaMA-Factory/saves/qwen3-4b/full/4B_1_0_1"  # 模型名称
dataset_path="/mnt/cfs/ssw/ljc/dataset_making/Data/ready_dataset/RLHF/0817/qwen3_dataset_full_0729_train_data_v2_after_vote_filtered_convert_grpo_tc.json"
output_dir="./output_ppo"  # 输出目录
wandb_api_key="8b7eb3957d2cf7157ab46fcf3e5b602cf2e7b24e"
swanlab_api_key="GFPjNmyR2K5Cog3C6N7uA"

# params
max_length=25000
max_completion_length=3000

# vLLM远程rollout配置
REMOTE_VLLM_HOST="10.217.16.121"  # 推理机
REMOTE_VLLM_PORT=8000

# 日志等级
export LOG_LEVEL=INFO
# 额外：rollout日志（可覆盖）
export ROLLOUT_LOG_LEVEL=${ROLLOUT_LOG_LEVEL:-INFO}
export ROLLOUT_LOG_PATH=${ROLLOUT_LOG_PATH:-/mnt/cfs/ssw/ljc/ms-swift/workSpace/GRPO/log/test.log}

# 配置GPU设备 (训练机器)  
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 提高CUDA内存利用率
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# 并行训练设置
export NPROC_PER_NODE=$nproc_per_node

# 分布式/网络相关（避免 IB/SHM 干扰，固定网卡，关闭调试日志）
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_BLOCKING_WAIT=1

WANDB_MODE=offline \
WANDB_API_KEY=$wandb_api_key \
SWANLAB_API_KEY=$swanlab_api_key \
swift rlhf \
    --rlhf_type grpo \
    --model $model_name \
    --model_type qwen3 \
    --train_type full \
    --agent_template qwen_en \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_tool_call_combined_cosine \
    --torch_dtype bfloat16 \
    --dataset $dataset_path \
    --max_length $max_length \
    --max_completion_length $max_completion_length \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir $output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --num_generations 8 \
    --deepspeed zero0 \
    --log_completions true \
    --report_to wandb \
    --beta 0.04 \
    --num_iterations 1 \
    --top_k 20 \
    --temperature 0.7 \
    --top_p 0.7 \
    --repetition_penalty 1.1 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host $REMOTE_VLLM_HOST \
    --vllm_server_port $REMOTE_VLLM_PORT \
    --vllm_server_timeout 30.0

# 关键修改说明：
# 1. 添加 --agent_template qwen_en 参数，确保使用Qwen专用的Agent Template
# 2. 这将使模型生成正确的工具调用格式：✿FUNCTION✿: 和 ✿ARGS✿:
# 3. 如果您的数据是中文，可以改为 --agent_template qwen_zh

# 可选的Agent Template选项：
# - qwen_en: 英文Qwen Agent Template  
# - qwen_zh: 中文Qwen Agent Template
# - hermes: Hermes风格的工具调用（JSON格式）
# - react_en: React英文模板
# - react_zh: React中文模板 