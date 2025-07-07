# Env: 8 * A100
# Max Length: 512000
# GPU Memory: 8 * 80GiB, Training Speed 150s/it
NPROC_PER_NODE=8 \
CELOSS_PARALLEL_SIZE=2048 \
swift sft \
    --model Qwen/QwQ-32B \
    --train_type lora \
    --dataset 'AI-ModelScope/LongAlpaca-12k' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --packing true \
    --rope_scaling yarn \
    --max_length 512000 \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --use_liger_kernel true \
    --save_only_model true \
    --deepspeed zero3_offload \
    --attn_impl flash_attn \
    --sequence_parallel_size 8
