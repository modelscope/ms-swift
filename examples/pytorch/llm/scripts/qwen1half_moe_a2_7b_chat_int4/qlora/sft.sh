# Experimental environment: A100
# 17GB GPU memory

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen1half-moe-a2_7b-chat-int4 \
    --sft_type lora \
    --output_dir output \
    --dataset blossom-math-zh \
    --train_dataset_sample -1 \
    --num_train_epochs 3 \
    --max_length 2048 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn true \
