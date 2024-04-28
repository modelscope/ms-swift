# Experiment env: A10, RTX3090/4090, A100
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --ckpt_dir "output/codeqwen1half-7b-chat-awq/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --use_flash_attn false \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --stream false \
    --merge_lora false \
