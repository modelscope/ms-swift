# Experimental environment: A100
# 52GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type deepseek-moe-16b-chat \
    --dataset damo-agent-mini-zh \
    --train_dataset_sample 20000 \
    --max_length 4096 \
    --gradient_checkpointing true \
    --eval_steps 100 \
    --use_flash_attn true \
    --output_dir output \
