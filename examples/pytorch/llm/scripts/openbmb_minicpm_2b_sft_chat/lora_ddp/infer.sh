# Experimental environment: A10
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --ckpt_dir "output/openbmb-minicpm-2b-sft-chat/vx_xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --max_length 2048 \
    --use_flash_attn false \
    --max_new_tokens 2048 \
    --do_sample false \
