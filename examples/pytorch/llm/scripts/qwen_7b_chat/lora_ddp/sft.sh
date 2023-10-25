# Experimental environment: 2 * A100
# 2 * 60GB GPU memory (use flash_attn)
# You need to install flash_attn or set gradient_checkpointing to True,
# otherwise it may result in an OOM (Out of Memory) error.
nproc_per_node=2

PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    llm_sft.py \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --model_revision master \
    --sft_type lora \
    --template_type chatml \
    --dtype bf16 \
    --output_dir output \
    --ddp_backend nccl \
    --dataset damo-agent-mini-zh \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 4096 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing false \
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
    --use_flash_attn true \
    --push_to_hub false \
    --hub_model_id qwen-7b-chat-lora \
    --hub_private_repo true \
    --hub_token 'your-sdk-token' \
