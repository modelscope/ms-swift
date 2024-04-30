# Experimental environment: A100
# 66GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type xverse-moe-a4_2b \
    --sft_type lora \
    --tuner_backend peft \
    --dtype fp16 \
    --dataset dureader-robust-zh \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 1024 \
    --check_dataset_strategy warning \
    --lora_dtype AUTO \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules DEFAULT \
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
