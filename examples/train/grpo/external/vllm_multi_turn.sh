# exp: https://github.com/modelscope/ms-swift/pull/4380#issuecomment-2992240961

# CUDA_VISIBLE_DEVICES=7 \
# swift rollout \
#     --model Qwen/Qwen2.5-3B-Instruct \
#     --use_async_engine true \
#     --multi_turn_scheduler math_tip_trick_multi_turn \
#     --max_turns 3

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
NPROC_PER_NODE=7 \
nohup swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-3B-Instruct \
    --train_type full \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --torch_dtype bfloat16 \
    --dataset AI-MO/NuminaMath-TIR#1000 \
    --split_dataset_ratio 0 \
    --max_completion_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 8 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 7 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --deepspeed zero2 \
    --log_completions true \
    --num_iterations 1 \
    --report_to tensorboard \
    --beta 0.04 \
    --multi_turn_scheduler math_tip_trick_multi_turn \
    --max_turns 3
