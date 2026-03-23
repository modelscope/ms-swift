SYSTEM_PROMPT="""You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."""

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-1.7B \
    --external_plugins examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py \
    --reward_funcs gsm8k_accuracy \
    --dataset 'modelscope/gsm8k' \
    --columns '{"answer": "solution"}' \
    --enable_thinking false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 10240 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --load_from_cache_file true \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --save_steps 10 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system "$SYSTEM_PROMPT" \
    --log_completions true \
    --num_iterations 1 \
    --beta 0.04 \
    --deepspeed zero2 \
    --report_to wandb \
    --loss_type real \
    --scale_rewards none \
    --log_entropy true
