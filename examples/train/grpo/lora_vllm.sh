# pip install math_verify # reward function
# pip install git+https://github.com/huggingface/trl.git
# GPU memory: 2 * 80GiB

MASTER_PORT=29501 \
CUDA_VISIBLE_DEVICES=0,1 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B \
    --reward_funcs accuracy format \
    --train_type lora \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 8192 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#1000' \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 16 \
    --temperature 0.9 \
    --deepspeed zero2 \
    --system 'examples/train/grpo/prompt.txt' \
    --log_completions true
