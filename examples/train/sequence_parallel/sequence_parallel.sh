# Env: 8 * A100
# Max Length: 65536
# GPU Memory: 8 * 40GiB, Training Speed 26s/it
NPROC_PER_NODE=8 \
CELOSS_PARALLEL_SIZE=2048 \
swift sft \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset 'AI-ModelScope/LongAlpaca-12k' \
    --load_from_cache_file true \
    --train_type lora \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --save_steps 50 \
    --max_length 65536 \
    --warmup_ratio 0.05 \
    --attn_impl flash_attn \
    --sequence_parallel_size 8 \
    --logging_steps 1 \
    --use_logits_to_keep false \
    --padding_free true \
