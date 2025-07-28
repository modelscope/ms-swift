# Env: 4 * A100
# Max Length: 65536
# GPU Memory: 4 * 38GiB, Training Speed 30s/it
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SEQUENCE_PARALLEL_IMPL=ring_attention \
RING_HEAD_STRIDE=2 \
swift sft \
    --model Qwen/Qwen2.5-3B-Instruct \
    --train_type full \
    --dataset 'AI-ModelScope/LongAlpaca-12k' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --packing true \
    --rope_scaling yarn \
    --max_length 65536 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output/Qwen2.5-3B-Instruct \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --sequence_parallel_size 4
