# 95GB GPU memory
# Experimental environment: 8 * 3090
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python src/llm_sft.py \
    --model_type qwen-7b-chat \
    --sft_type full \
    --template_type chatml \
    --dtype bf16 \
    --output_dir runs \
    --dataset alpaca-en,alpaca-zh \
    --dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 1024 \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 1 \
    --warmup_ratio 0.03 \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn false \
    --push_to_hub false \
    --hub_model_id qwen-7b-chat-full \
    --hub_private_repo true \
    --hub_token 'your-sdk-token' \
