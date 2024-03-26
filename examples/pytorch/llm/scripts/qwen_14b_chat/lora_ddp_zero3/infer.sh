# Experimental environment: 2 * 3090

CUDA_VISIBLE_DEVICES=0,1 \
swift infer \
    --ckpt_dir "output/qwen-14b-chat/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --use_flash_attn false \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora false \
