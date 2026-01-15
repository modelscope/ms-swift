# 2*70GiB
# losses: plugin/loss.py
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model Qwen/Qwen3-VL-Reranker-8B \
    --task_type generative_reranker \
    --loss_type pointwise_reranker \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --learning_rate 5e-6 \
    --target_modules all-linear \
    --dataset swift/TextCaps:rerank \
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
    --max_length 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --warmup_ratio 0.05 \
    --dataloader_drop_last true \
    --deepspeed zero2
