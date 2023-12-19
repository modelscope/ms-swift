# Experimental environment: A10
# 8GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --ckpt_dir "phi2-3b/vx_xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --max_length 2048 \
    --use_flash_attn false \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --top_p 0.7 \
    --repetition_penalty 1.05 \
    --merge_lora_and_save false \
