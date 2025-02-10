# pip install math_verify # reward function
# pip install git+https://github.com/huggingface/trl.git # trl >=0.15.0.dev0
# GPU memory: 80GiB
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --reward_funcs accuracy format \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#2000' \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 0.7 \
    --system 'examples/train/grpo/prompt.txt'
