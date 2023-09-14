# Experimental environment: V100(16GB)
# 14GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
python src/llm_sft.py \
    --model_type chatglm2-6b \
    --sft_type lora \
    --template_type chatglm2 \
    --dtype bf16 \
    --output_dir runs \
    --dataset advertise-gen \
    --dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 2048 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0. \
    --lora_target_modules query_key_value \
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
    --push_to_hub false \
    --hub_model_id chatglm2-6b-lora \
    --hub_private_repo true \
    --hub_token 'your-sdk-token' \
