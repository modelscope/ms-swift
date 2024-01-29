# Experimental environment: A10, 3090
# 10GB GPU memory
nproc_per_node=4

PYTHONPATH=../../.. \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    llm_sft.py \
    --model_id_or_path ZhipuAI/chatglm3-6b-32k \
    --model_revision master \
    --sft_type full \
    --train_dataset_mix_ratio 2.0 \
    --tuner_backend swift \
    --dtype AUTO \
    --output_dir output \
    --dataset ms-agent \
    --train_dataset_sample -1 \
    --num_train_epochs 2 \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --self_cognition_sample 3000 \
    --model_name 小灰灰 \
    --model_author 陶白白 \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --push_to_hub false \
    --hub_model_id chatglm3-6b-32k-qlora \
    --hub_private_repo true \
    --hub_token 'your-sdk-token'