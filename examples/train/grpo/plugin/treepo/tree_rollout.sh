# This script is a example for multi-turn training with tree-rollout.
# Regarding parameter configuration, currently tree_rollout, acting as the inference side, cannot receive relevant training parameters. Please note the following:
#   1.Ensure that max_tree_width in tree_rollout is equal to num_generations.
#   2.If DP (Data Parallelism) is enabled during the rollout stage, ensures that data within the same group is allocated to the same inference device.
#     For example: If generation_batch_size(per_device_batch_size * gradient_accumulation_steps * num_processes) = 32 and num_generations = 8,
#     then the rollout DP num should equal 4/2/1.
# For more details on tool invocation, dialogue termination criteria, and other logic, please refer to the TreeRolloutScheduler implementation.

# First: Run swift rollout to deploy rollout server
CUDA_VISIBLE_DEVICES=0 \
swift rollout \
    --model Qwen/Qwen2.5-0.5B \
    --vllm_use_async_engine true \
    --external_plugins examples/train/grpo/plugin/treepo/tree_rollout_plugin.py \
    --multi_turn_scheduler tree_rollout_scheduler \
    --max_turns 6


# Second: Run swift rlhf to train GRPO model
CUDA_VISIBLE_DEVICES=1 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-0.5B \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type full \
    --torch_dtype bfloat16 \
    --external_plugins examples/train/grpo/plugin/treepo/tree_rollout_plugin.py \
    --dataset AI-MO/NuminaMath-TIR#1000 \
    --split_dataset_ratio 0 \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --log_completions true \
    --num_iterations 1 \
    --beta 0.04
