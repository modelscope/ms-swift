#!/bin/bash
# 10.217.16.160   wangxiang4-bce-a800-1-0 训练机
# 10.217.16.121   wangxiang4-bce-a800-2-0 推理机

nproc_per_node=8  # 使用的GPU数量，根据你的硬件调整
model_name="/mnt/cfs/ssw/ljc/LLaMA-Factory/models/Qwen3-4B"  # 模型名称
output_dir="./output_ppo"  # 输出目录

# vLLM远程rollout配置
REMOTE_VLLM_HOST="10.217.16.121"  # 推理机
REMOTE_VLLM_PORT=8000

# 配置GPU设备 (训练机器)  
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 提高CUDA内存利用率
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# 并行训练设置
export NPROC_PER_NODE=$nproc_per_node

# 启动远程vLLM服务器 (在远程机器上执行)
echo "请确保在远程机器 $REMOTE_VLLM_HOST 上启动vLLM服务:"
echo ""
echo "   权重更新机制说明:"
echo "   - vLLM启动时使用基线模型权重: $model_name"
echo "   - PPO训练中策略权重会实时同步到远程vLLM服务器"
echo "等待vLLM服务启动完成后，按任意键继续..."
read -n 1

# 执行PPO训练命令
swift rlhf \
    --rlhf_type ppo \
    --model $model_name \
    --model_type qwen3 \
    --check_model false \
    --train_type full \
    --async_generate true \
    --dataset /mnt/cfs/ssw/ljc/LLaMA-Factory/data/qwen3_dataset_full_0729_train_data_v2_after_vote_filtered_convert_ppo.json \
    --split_dataset_ratio 0.01 \
    --dataset_num_proc 4 \
    --torch_dtype bfloat16 \
    --max_length 25000 \
    --max_completion_length 2500 \
    --truncation_strategy delete \
    --padding_side left \
    --num_ppo_epochs 4 \
    --kl_coef 0.05 \
    --cliprange 0.2 \
    --vf_coef 0.1 \
    --cliprange_value 0.2 \
    --gamma 1.0 \
    --lam 0.95 \
    --local_rollout_forward_batch_size 64 \
    --whiten_rewards false \
    --sync_ref_model true \
    --ref_model_sync_steps 50 \
    --ref_model_mixup_alpha 0.3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --log_completions true \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir $output_dir \
    --save_only_model true \
    --deepspeed zero3 \
    --dataloader_num_workers 0 \
    --report_to wandb \
    --run_name "ppo_remote_vllm_$(date +%Y%m%d_%H%M%S)" \
    --temperature 0.6 \
    --top_p 0.6 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host $REMOTE_VLLM_HOST \
    --vllm_server_port $REMOTE_VLLM_PORT \
    --vllm_server_timeout 300.0 \

    # --ref_model_sync_steps 50 设为steps的5%
    # --ref_model_mixup_alpha 0.3  单次更新 30% 权重
    # PPO要求一个价值模型，作为状态价值计算，因此不能使用奖励函数 --reward_funcs
    # PPO会自动使用主模型创建价值头，无需单独指定价值模型
    # 无--reward_model 情况下 trl使用 AutoModelForCausalLMWithValueHead 包装主模型

    # --reward_model orm://CombinedCosineReward \