# Env: 4 * A100
# Max Length: 256000
# GPU Memory: 4 * 42GiB, Training Speed 43s/it
NPROC_PER_NODE=4 \
CELOSS_PARALLEL_SIZE=2048 \
SEQUENCE_PARALLEL_IMPL=ring_attention \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/LongAlpaca-12k' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --packing true \
    --rope_scaling yarn \
    --max_length 256000 \
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
    --sequence_parallel_size 4
