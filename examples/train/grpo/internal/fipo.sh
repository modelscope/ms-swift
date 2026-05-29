CUDA_VISIBLE_DEVICES=2 \
swift rollout \
    --model Qwen/Qwen2.5-1.5B-Instruct

# 2 GPUS for sequence parallel
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset 'AI-MO/NuminaMath-TIR'  \
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
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_total_limit 3 \
    --save_steps 500 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --num_generations 8 \
    --temperature 1.0 \
    --system """You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}.""" \
    --log_completions true \
    --num_iterations 3 \
    --padding_free true \
    --sequence_parallel_size 2 \
    --attn_impl flash_attn \
    --beta 0 \
    --dynamic_sample true \
    --loss_type fipo \
    --delta 10.0 \
    --epsilon_high 0.28 \
    --fipo_decay_rate 32 \
    --fipo_clip_range 0.2 \
    --fipo_clip_high_only false \
    --fipo_safety_threshold 3.0
