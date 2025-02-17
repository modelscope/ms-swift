# pip install math_verify # reward function
# pip install "trl>=0.15"
# GPU memory: 80GiB
# You can set `--reward_model` to use a reward model to provide rewards.
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --reward_funcs accuracy format \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#1000' \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
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
    --num_generations 4 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt'
