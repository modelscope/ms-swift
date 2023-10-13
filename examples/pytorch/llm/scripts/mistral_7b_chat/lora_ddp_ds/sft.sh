# Experimental environment: 2 * A10
# 2 * 21GB GPU memory
nproc_per_node=2

PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    src/llm_sft.py \
    --model_type mistral-7b-chat \
    --sft_type lora \
    --template_type llama \
    --dtype bf16 \
    --output_dir output \
    --ddp_backend nccl \
    --dataset leetcode-python-en \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 4096 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0. \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0. \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --push_to_hub false \
    --hub_model_id mistral-7b-chat-qlora \
    --hub_private_repo true \
    --hub_token 'your-sdk-token' \
    --deepspeed_config_path 'ds_config/zero2.json' \
    --only_save_model true \
