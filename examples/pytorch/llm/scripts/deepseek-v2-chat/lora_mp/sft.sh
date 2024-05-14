# Experimental environment: 8*A100
# 8*80GB GPU memory
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model_type deepseek-v2-chat \
    --sft_type lora \
    --tuner_backend peft \
    --dtype bf16 \
    --output_dir output \
    --ddp_backend nccl \
    --dataset alpaca-zh#5000 \
    --num_train_epochs 1 \
    --max_length 1024 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_dtype AUTO \
    --lora_target_modules DEFAULT \
    --gradient_checkpointing true \
    --use_flash_attn true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16\
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --device_map_config_path scripts/deepseek-v2-chat/lora_ddp_ds3/deepseek2_device_map.json
