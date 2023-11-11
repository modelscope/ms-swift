# Experimental environment: 2 * 3090
# 2 * 20GB GPU memory
nproc_per_node=1

PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
python llm_sft.py \
    --model_id_or_path ZhipuAI/chatglm3-6b-32k \
    --model_revision master \
    --sft_type full \
    --tuner_backend swift \
    --template_type chatglm3 \
    --dtype bf16 \
    --output_dir output \
    --ddp_backend nccl \
    --dataset dureader-robust-zh \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 512 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules AUTO \
    --gradient_checkpointing true \
    --batch_size 16 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --push_to_hub false \
    --hub_model_id chatglm3-6b-32k-lora \
    --hub_private_repo true \
    --hub_token 'your-sdk-token' \
    --deepspeed_config_path 'ds_config/zero2.json' \
    --only_save_model true \
