# Experimental environment: A10
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --ckpt_dir "output/yuan2-2b-janus-instruct/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --use_flash_attn false \
    --max_new_tokens 2048 \
    --do_sample false \
    --merge_lora false \
