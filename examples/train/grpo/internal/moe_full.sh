# 8*80G

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 16384 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset AI-MO/NuminaMath-TIR#1000 \
    --max_length 12000 \
    --max_completion_length 8192 \
    --overlong_filter true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 16 \
    --temperature 1.0 \
    --deepspeed zero3_offload \
    --log_completions true \
    --sleep_level 1 \
    --report_to tensorboard swanlab \
    --num_iterations 1 \
    --beta 0.001 \
    --move_model_batches 10
