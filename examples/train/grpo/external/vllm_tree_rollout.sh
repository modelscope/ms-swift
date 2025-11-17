# This script is a example for multi-turn training with tree-rollout

CUDA_VISIBLE_DEVICES=2 \
swift rollout \
    --model Qwen2.5-0.5B


CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen2.5-0.5B \
    --reward_funcs format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset AI-MO/NuminaMath-TIR#1000 \
    --split_dataset_ratio 0 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --deepspeed zero2 \
    --gradient_checkpointing false \
    --log_completions true \
    --num_iterations 1 \
    --beta 0.04 \
    --tree_rollout true \
    --multi_turn_scheduler tree_rollout_scheduler \
    --max_tree_deep 4