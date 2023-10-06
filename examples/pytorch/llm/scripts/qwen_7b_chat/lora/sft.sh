# Experimental environment: A100
# 60GB GPU memory (use flash_attn)
# You need to install flash_attn or set gradient_checkpointing to True,
# otherwise it may result in an OOM (Out of Memory) error.
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python src/llm_sft.py \
    --model_type qwen-7b-chat \
    --sft_type lora \
    --template_type chatml \
    --dtype bf16 \
    --output_dir output \
    --dataset damo-agent-mini-zh \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 4096 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0. \
    --lora_target_modules ALL \
    --gradient_checkpointing false \
    --batch_size 1 \
    --weight_decay 0. \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn true \
    --push_to_hub false \
    --hub_model_id qwen-7b-chat-lora \
    --hub_private_repo true \
    --hub_token 'your-sdk-token' \
