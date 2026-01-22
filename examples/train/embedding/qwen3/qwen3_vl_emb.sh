# 2*30GiB
CUDA_VISIBLE_DEVICES=0,1 \
INFONCE_TEMPERATURE=0.1 \
NPROC_PER_NODE=2 \
swift sft \
    --model Qwen/Qwen3-VL-Embedding-8B \
    --task_type embedding \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --learning_rate 5e-5 \
    --target_modules all-linear \
    --dataset swift/TextCaps:emb \
    --attn_impl flash_attn \
    --padding_free true \
    --torch_dtype bfloat16 \
    --load_from_cache_file true \
    --split_dataset_ratio 0.02 \
    --eval_strategy steps \
    --output_dir output \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --num_train_epochs 1 \
    --max_length 8192 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --warmup_ratio 0.05 \
    --loss_type infonce \
    --dataloader_drop_last true \
    --deepspeed zero2
