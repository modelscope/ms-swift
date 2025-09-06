# Env: 4 * A100
# Max Length: 65536
# GPU Memory: 8 * 48GiB, Training Speed 21.84s/it]
NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset 'AI-ModelScope/LongAlpaca-12k' \
    --train_type lora \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --target_modules all-linear \
    --load_from_cache_file false \
    --gradient_accumulation_steps 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --save_steps 30 \
    --max_length 65536 \
    --warmup_ratio 0.05 \
    --attn_impl flash_attn \
    --sequence_parallel_size 8 \
    --logging_steps 1 \
    --use_logits_to_keep false \
    --model_author swift \
    --model_name kikibot \
    --padding_free true \
