# zero2: 70GiB
IMAGE_MAX_TOKEN_NUM=1024 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#20000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --attn_impl flash_attn \
    --padding_free true \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --router_aux_loss_coef 1e-3 \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --use_liger_kernel true \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4
