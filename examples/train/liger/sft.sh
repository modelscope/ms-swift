# test env: 4 * A100
# Using use_liger_kernel and packing: 4 * 42GB, 1 hour 35 minutes
# Not using use_liger_kernel: 4 * 54GB, 1 hour 40 minutes
# Not using use_liger_kernel and packing: 4 * 52GB, 3 hours 30 minutes

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2.5-7B \
    --train_type full \
    --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT#10000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output/Qwen2.5-7B \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --packing true \
    --use_liger_kernel true
