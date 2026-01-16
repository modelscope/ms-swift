# 8 * 80 G (6 for training, 2 for rollout)
# CUDA_VISIBLE_DEVICES=6,7 \
# swift rollout \
#     --model Qwen/Qwen2.5-7B-Instruct \
#     --data_parallel_size 2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset AI-MO/NuminaMath-TIR#1000 \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --max_completion_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --deepspeed zero2 \
    --log_completions true \
    --num_iterations 1 \
    --report_to tensorboard wandb \
    --beta 0.04
