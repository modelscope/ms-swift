#!/bin/bash
# 10.217.16.160   wangxiang4-bce-a800-1-0 训练机
# 10.217.16.121   wangxiang4-bce-a800-2-0 推理机

# hard settings
# nproc_per_node=8  # 使用的GPU数量，根据你的硬件调整
nproc_per_node=2 
model_name="/mnt/cfs/ssw/ljc/LLaMA-Factory/saves/qwen3-4b/full/4B_1_0_1"  # 模型名称
dataset_path="/mnt/cfs/ssw/ljc/dataset_making/Data/ready_dataset/RLHF/0817/grpo_tc.json"
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
# mkdir -p "$(dirname "$ROLLOUT_LOG_PATH")" 2>/dev/null || true
# if ! touch "$ROLLOUT_LOG_PATH" 2>/dev/null; then
# 	mkdir -p "$HOME/.ms-swift" 2>/dev/null || true
# 	export ROLLOUT_LOG_PATH="$HOME/.ms-swift/rollout.log"
# 	touch "$ROLLOUT_LOG_PATH" 2>/dev/null || echo "无法创建日志文件：$ROLLOUT_LOG_PATH"
# fi

# echo "rollout日志: $ROLLOUT_LOG_PATH (level=$ROLLOUT_LOG_LEVEL)"

# 配置GPU设备 (训练机器)  
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1

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
# 训练器调试
export LOG_LEVEL=DEBUG
export SWIFT_DEBUG=true

# 添加梯度稳定性参数
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

WANDB_MODE=offline \
WANDB_API_KEY=$wandb_api_key \
SWANLAB_API_KEY=$swanlab_api_key \
swift rlhf \
    --rlhf_type grpo \
    --model $model_name \
    --model_type qwen3 \
    --train_type full \
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
    --max_grad_norm 0.8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir $output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --num_generations 2 \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --beta 0.04 \
    --num_iterations 1 \
    --top_k 20 \
    --temperature 0.6 \
    --top_p 0.6 \
    --repetition_penalty 1.1 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host $REMOTE_VLLM_HOST \
    --vllm_server_port $REMOTE_VLLM_PORT \
    --vllm_server_timeout 30.0 \
    --gradient_checkpointing false \
    --max_turns 1 \
    --async_generate false

# deepspeed: 默认为None。可以设置为'zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'来使用ms-swift内置的deepspeed配置文件
# 训练集路径
# /mnt/cfs/ssw/ljc/LLaMA-Factory/data/qwen3_dataset_full_0729_train_data_v2_after_vote_filtered_convert_ppo.json
# /mnt/cfs/ssw/wx/code/dev/chat_agent_data_process/Data/训练数据/v2/val_data_v2_del_model_output.json
# test
# /mnt/cfs/ssw/ljc/dataset_making/Data/ready_dataset/RLHF/0817/test_tc.json
    # 解决"None of the inputs have requires_grad=True"警告:
    # - zero2比zero3在RLHF中更稳定
    # - 禁用gradient_checkpointing避免rollout阶段梯度冲突
# beta: KL正则项系数，默认为`None`，即`simpo`算法默认为`2.`，GRPO默认为`0.04`，GKD默认为0.5，其他算法默认为`0.1`。具体参考[文档](./人类对齐.md)。
# multi_turn_scheduler: 多轮GRPO参数, 传入对应的plugin名称, 同时在plugin/multi_turn.py中添加好对应的实现。
# max_turns: 多轮GRPO的轮数上限。默认为None，不做限制。
