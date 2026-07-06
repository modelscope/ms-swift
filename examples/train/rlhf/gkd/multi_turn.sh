# 1 * 73GiB, multi-turn GKD with math_tip_trick scheduler
NPROC_PER_NODE=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3.5-0.8B \
    --teacher_model Qwen/Qwen3.5-2B \
    --tuner_type lora \
    --dataset 'AI-ModelScope/NuminaMath-TIR#2000' \
    --lmbda 1 \
    --beta 0.5 \
    --temperature 1.0 \
    --torch_dtype bfloat16 \
    --max_steps 200 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --save_steps 200 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 2048 \
    --max_completion_length 1024 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --teacher_deepspeed zero3_offload \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_max_model_len 4096 \
    --sleep_level 1 \
    --enable_thinking false \
    --multi_turn_scheduler math_tip_trick \
    --max_turns 2 \
    --truncation_strategy delete
