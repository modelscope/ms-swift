CUDA_VISIBLE_DEVICES=2 \
swift rollout \
    --model Qwen/Qwen3-1.7B


NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-1.7B \
    --dataset 'AI-MO/NuminaMath-TIR#5000'  \
    --enable_thinking false \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --load_from_cache_file true \
    --max_completion_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-6 \
    --gradient_accumulation_steps 1 \
    --save_total_limit 2 \
    --save_steps 500 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 0.6 \
    --system """You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}.""" \
    --log_completions true \
    --num_iterations 1 \
    --beta 0.001 \
    --loss_type real \
    --deepspeed zero2
