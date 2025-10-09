# 4*80G GPU

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-72B-Instruct \
    --train_type lora \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.55 \
    --vllm_max_model_len 2048 \
    --vllm_tensor_parallel_size 4 \
    --dataset AI-MO/NuminaMath-TIR#10000 \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --learning_rate 1e-6 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_completion_length 1024 \
    --reward_funcs accuracy format \
    --num_generations 4 \
    --system examples/train/grpo/prompt.txt \
    --deepspeed zero3_offload \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 80 \
    --log_completions true \
    --async_generate false \
    --move_model_batches 16 \
    --offload_optimizer true \
    --offload_model true \
    --sleep_level 1
