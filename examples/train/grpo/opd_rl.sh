SYSTEM_PROMPT="""You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."""

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-2B \
    --teacher_model Qwen/Qwen3.5-9B \
    --enable_thinking false \
    --tuner_type lora \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --sleep_level 1 \
    --dataset 'modelscope/gsm8k' \
    --torch_dtype bfloat16 \
    --num_generations 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 10 \
    --max_length 2048 \
    --max_completion_length 2048 \
    --deepspeed zero2 \
    --warmup_ratio 0.1 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --attn_impl flash_attn \
    --log_completions true \
    --log_rollout_offpolicy_metrics true \
    --report_to tensorboard swanlab
