# Experimental environment: 2 * A10
# 2 * 13GB GPU memory (not use flash_attn)
nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
swift sft \
    --model_id_or_path qwen/Qwen-VL-Chat-Int4 \
    --model_revision master \
    --sft_type lora \
    --tuner_backend swift \
    --template_type chatml \
    --dtype fp16 \
    --output_dir output \
    --ddp_backend nccl \
    --dataset coco-en \
    --train_dataset_sample 20000 \
    --num_train_epochs 1 \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules c_attn attn.c_proj \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn false \
    --push_to_hub false \
    --hub_model_id qwen-vl-chat-qlora \
    --hub_private_repo true \
    --hub_token 'your-sdk-token' \
    --deepspeed_config_path 'ds_config/zero2.json' \
    --only_save_model true \
