# Experimental environment: 4 * A100
# 4 * 65GB GPU memory
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --ckpt_dir "output/dbrx-instruct/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --use_flash_attn true \
    --temperature 0.3 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora false \
