# 2 * 60GiB
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_PORT=29605 \
NPROC_PER_NODE=2 \
swift sft \
    --model google/diffusiongemma-26B-A4B-it \
    --dataset 'sapientinc/sudoku-extreme-1k' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --loss_scale ignore_empty_think \
    --gradient_checkpointing false \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --dataloader_num_workers 4

CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --load_data_args true \
    --enable_thinking false
