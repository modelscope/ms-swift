# 4 * 26GB
# This example is just a demo showing how to add new_special_tokens.
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'swift/new_special_tokens' \
    --split_dataset_ratio 0.01 \
    --new_special_tokens 'examples/train/new_special_tokens/tokens.txt' \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --padding_free true \
    --attn_impl flash_attn \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --modules_to_save embed_tokens lm_head \
    --gradient_accumulation_steps 1 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2
