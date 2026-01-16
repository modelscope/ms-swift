# 18GB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --train_type lora \
    --dataset 'swift/DeepSeek-R1-Qwen3-8B-Distill#1800' \
              'swift/self-cognition:empty_think#600' \
    --loss_scale ignore_empty_think \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --load_from_cache_file false \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --use_liger_kernel true \
    --model_author swift \
    --model_name swift-robot
