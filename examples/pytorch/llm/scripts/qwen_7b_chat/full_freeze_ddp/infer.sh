# Experimental environment: A100
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --ckpt_dir "output/qwen-7b-chat/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --use_flash_attn true \
