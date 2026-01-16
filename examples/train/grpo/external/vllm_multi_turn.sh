# Exp: https://github.com/modelscope/ms-swift/pull/5307#issuecomment-3219803922
# Before running this script, please run the following `swift rollout` script first
# This script is a example for multi-turn training with dynamic num of rollout outputs
# which means a trajectory of multi turn rollout is split into multiple data
#       see details in thinking_tips_scheduler
# NOTE: for same trajectory, the reward is supported to be the same,
#       here we use the last turn data of each trajectory to compute accuracy reward
#       see details in thinking_tips reward function

# CUDA_VISIBLE_DEVICES=0 \
# swift rollout \
#     --model Qwen/Qwen3-1.7B \
#     --vllm_use_async_engine true \
#     --multi_turn_scheduler thinking_tips_scheduler \
#     --vllm_max_model_len 32768 \
#     --vllm_gpu_memory_utilization 0.8 \
#     --max_turns 3

CUDA_VISIBLE_DEVICES=1,2 \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-1.7B \
    --train_type full \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs thinking_tips \
    --loss_scale last_round \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --vllm_server_pass_dataset true \
    --torch_dtype bfloat16 \
    --dataset AI-MO/NuminaMath-TIR#10000 \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --max_completion_length 8192 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --steps_per_generation 8 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --deepspeed zero2 \
    --log_completions true \
    --log_entropy true \
    --importance_sampling_level sequence \
    --top_entropy_quantile 0.2 \
    --num_iterations 1 \
    --report_to tensorboard swanlab
