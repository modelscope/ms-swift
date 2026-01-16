# ms-swift>=3.12
OMP_NUM_THREADS=14 \
MAX_PIXELS=1003520 \
swift export \
    --model Qwen/Qwen2.5-Omni-3B \
    --dataset 'tany0699/garbage265#20000' \
    --task_type seq_cls \
    --num_labels 265 \
    --problem_type single_label_classification \
    --split_dataset_ratio 0.01 \
    --dataset_num_proc 16 \
    --to_cached_dataset true \
    --output_dir ./seq_cls_cached_dataset


# 18GiB
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-Omni-3B \
    --tuner_type lora \
    --cached_dataset 'seq_cls_cached_dataset/train' \
    --cached_val_dataset 'seq_cls_cached_dataset/val' \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 2 \
    --packing true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --task_type seq_cls \
    --num_labels 265 \
    --problem_type single_label_classification \
    --use_chat_template true \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output/Qwen2.5-Omni-3B \
    --attn_impl flash_attn

# Use the validation set
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift infer \
    --adapters output/Qwen2.5-Omni-3B/vx-xxx/checkpoint-xxx \
    --load_data_args true \
    --attn_impl flash_attn
