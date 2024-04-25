# Experimental environment: 4*A100
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --ckpt_dir "output/qwen1half-110b-chat/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --use_flash_attn true \
    --max_new_tokens 2048 \
    --temperature 0.5 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora false \
