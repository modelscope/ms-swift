# A800 * 8
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-1.5B \
    --reward_funcs accuracy format \
    --use_lmdeploy true \
    --lmdeploy_cache_max_entry_count 0.7 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset AI-MO/NuminaMath-TIR#5000 \
    --max_completion_length 1536 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 32 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --num_iterations 2 \
    --move_model_batches 16 \
    --offload_optimizer true \
    --offload_model true \
    --gc_collect_after_offload true