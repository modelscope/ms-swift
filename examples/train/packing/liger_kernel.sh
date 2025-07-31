# Env: 4 * A100
# https://github.com/modelscope/ms-swift/blob/main/examples/train/megatron/long_text.sh
# Max Length: 16K
# GPU Memory: 4 * 42GB, Training Speed 10s/it
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2.5-7B \
    --train_type full \
    --dataset 'AI-ModelScope/LongAlpaca-12k' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --packing true \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --max_length 16384 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output/Qwen2.5-7B \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attn
