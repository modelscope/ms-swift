# 4 * 50GiB
pip install transformers math_verify trl -U

MAX_PIXELS=1003520 \
NPROC_PER_NODE=4 \
ENABLE_AUDIO_OUTPUT=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-Omni-7B \
    --reward_funcs external_r1v_acc format \
    --reward_weights 1 0.5 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset lmms-lab/multimodal-open-r1-8k-verified#1000 \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1. \
    --top_p 0.99 \
    --top_k 50 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true
