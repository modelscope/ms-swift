# Experimental environment: 4 * A100
# 4 * 74GB GPU memory
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type dbrx-instruct \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --template_type AUTO \
    --dtype bf16 \
    --output_dir output \
    --ddp_backend nccl \
    --dataset blossom-math-zh \
    --num_train_epochs 1 \
    --max_length 1024 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --lora_dtype AUTO \
    --gradient_checkpointing false \
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
    --use_flash_attn true
