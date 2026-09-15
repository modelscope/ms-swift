#!/bin/bash
# 调试模式的GRPO训练脚本

# 基础配置
model_name="/mnt/cfs/ssw/ljc/LLaMA-Factory/models/Qwen3-4B"
output_dir="./debug_output"
wandb_api_key="8b7eb3957d2cf7157ab46fcf3e5b602cf2e7b24e"

# 简化配置用于快速调试
max_length=512
max_completion_length=128

# 配置GPU设备
export CUDA_VISIBLE_DEVICES=0  # 只使用一个GPU进行调试

# Python调试环境变量
export PYTHONPATH="/mnt/cfs/ssw/ljc/ms-swift:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1  # 避免生成.pyc文件
export PYTHONUNBUFFERED=1        # 确保输出实时显示

echo "===== 调试模式启动 ====="
echo "按回车键开始调试..."
read

# 启动调试模式的GRPO训练
WANDB_API_KEY=$wandb_api_key \
python -u -m pdb /mnt/cfs/ssw/ljc/ms-swift/swift/cli/rlhf.py \
    --rlhf_type grpo \
    --model $model_name \
    --model_type qwen3 \
    --train_type full \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_combined_cosine \
    --torch_dtype bfloat16 \
    --dataset /mnt/cfs/ssw/ljc/LLaMA-Factory/data/qwen3_dataset_full_0729_train_data_v2_after_vote_filtered_convert_ppo.json \
    --max_length $max_length \
    --max_completion_length $max_completion_length \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 1 \
    --eval_steps 5 \
    --save_steps 5 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --output_dir $output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --num_generations 2 \
    --deepspeed zero2 \
    --log_completions true \
    --report_to none \
    --beta 0.001 \
    --num_iterations 1 \
    --temperature 0.6 \
    --top_p 0.6 \
    --split_dataset_ratio 0.001  # 使用极少量数据进行调试 