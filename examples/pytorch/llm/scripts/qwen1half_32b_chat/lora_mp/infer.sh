# Experimental environment: V100, A10, 3090
# 66GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --ckpt_dir "output/qwen1half-32b-chat/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --use_flash_attn true \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora false \
