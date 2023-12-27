# Experimental environment: 2 * A100
# 2 * 45GB
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
python llm_sft.py \
    --model_type cogagent-chat \
    --sft_type lora \
    --tuner_backend swift \
    --dtype fp16 \
    --output_dir output \
    --dataset capcha-images \
    --train_dataset_sample -1 \
    --num_train_epochs 2 \
    --max_length 1024 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --gradient_checkpointing false \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10
    --push_to_hub false \
    --hub_model_id cogagent-chat-lora \
    --hub_private_repo true \
    --hub_token 'your-sdk-token' \
