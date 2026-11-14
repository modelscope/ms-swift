#!/bin/bash
# 自动价值头配置示例
# 10.217.16.160   wangxiang4-bce-a800-1-0 训练机
# 10.217.16.121   wangxiang4-bce-a800-2-0 推理机

nproc_per_node=8  # 使用的GPU数量，根据你的硬件调整
model_name="/mnt/cfs/ssw/ljc/LLaMA-Factory/models/Qwen3-4B"  # 模型名称
output_dir="./output_ppo_auto_value"  # 输出目录

# vLLM远程rollout配置
REMOTE_VLLM_HOST="10.217.16.121"  # 推理机
REMOTE_VLLM_PORT=8000

# 配置GPU设备 (训练机器)  
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 提高CUDA内存利用率
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# 并行训练设置
export NPROC_PER_NODE=$nproc_per_node

echo "===== 自动价值头配置说明 ====="
echo "当不指定reward_model或value_model时："
echo "1. Swift会自动使用AutoModelForCausalLMWithValueHead"
echo "2. 在主模型基础上添加价值头（线性层）"
echo "3. 策略和价值函数共享语言模型的表示层"
echo "4. 更节省GPU内存，训练更稳定"
echo ""
echo "等待确认后继续..."
read -n 1

# 执行PPO训练命令 - 自动价值头配置
swift rlhf \
    --rlhf_type ppo \
    --model $model_name \
    --model_type qwen3 \
    --check_model false \
    \
    # ===== 关键：不指定reward_model让系统自动创建价值头 ===== \
    # --reward_model 留空，Swift会自动处理
    # --value_model 留空，Swift会自动处理
    \
    --train_type full \
    \
    # ===== 远程vLLM Rollout配置 ===== \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host $REMOTE_VLLM_HOST \
    --vllm_server_port $REMOTE_VLLM_PORT \
    --vllm_server_timeout 300.0 \
    --async_generate true \
    \
    # ===== 数据集配置 ===== \
    --dataset /mnt/cfs/ssw/ljc/LLaMA-Factory/data/qwen3_dataset_full_0729_train_data_v2_after_vote_filtered_convert_ppo.json \
    --split_dataset_ratio 0.01 \
    --dataset_num_proc 4 \
    \
    # ===== 模型配置 ===== \
    --torch_dtype bfloat16 \
    --max_length 25000 \
    --max_completion_length 2500 \
    --truncation_strategy delete \
    --padding_side left \
    \
    # ===== PPO特定参数 ===== \
    --num_ppo_epochs 4 \
    --kl_coef 0.05 \
    --cliprange 0.2 \
    --vf_coef 0.1 \
    --cliprange_value 0.2 \
    --gamma 1.0 \
    --lam 0.95 \
    --local_rollout_forward_batch_size 64 \
    --whiten_rewards false \
    \
    # ===== 权重同步优化配置 ===== \
    --sync_ref_model true \
    --ref_model_sync_steps 50 \
    --ref_model_mixup_alpha 0.3 \
    \
    # ===== 训练参数 ===== \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    \
    # ===== 日志和保存设置 ===== \
    --log_completions true \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir $output_dir \
    --save_only_model true \
    \
    # ===== 分布式训练加速 ===== \
    --deepspeed zero3 \
    --dataloader_num_workers 0 \
    --report_to wandb \
    --run_name "ppo_auto_value_head_$(date +%Y%m%d_%H%M%S)" \
    --temperature 0.6 \
    --top_p 0.6

echo ""
echo "===== 自动价值头配置完成 ====="
echo "模型会自动创建价值头，无需手动指定value_model" 