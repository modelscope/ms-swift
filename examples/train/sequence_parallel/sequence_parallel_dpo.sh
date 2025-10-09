# Env: 4 * A100
# GPU Memory: 4 * 25GiB, Training Speed 14s/it
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --train_type full \
    --dataset swift/RLAIF-V-Dataset \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output/Qwen2.5-VL-3B-Instruct \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --use_liger_kernel true \
    --sequence_parallel_size 4
