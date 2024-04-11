# Experimental environment: 4*A100
# 4 * 55GB GPU memory
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --ckpt_dir "output/c4ai-command-r-plus/vx-xxx/checkpoint-xx" \
    --load_dataset_config true \
    --load_args_from_ckpt_dir true \
    --temperature 0.3 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora false \
