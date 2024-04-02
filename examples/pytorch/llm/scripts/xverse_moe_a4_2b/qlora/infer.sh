# Experimental environment: 3090
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --ckpt_dir "output/xverse-moe-a4_2b/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora false \
