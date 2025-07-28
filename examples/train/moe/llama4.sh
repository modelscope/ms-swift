# Manually select `target_modules` to avoid 'all-linear' selecting 'router'
NPROC_PER_NODE=4 \
USE_HF=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --dataset 'linxy/LaTeX_OCR:full#5000' \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_regex '^(language_model).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$' \
    --freeze_vit true \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --dataloader_num_workers 4
