# Experimental environment: 2*A100

CUDA_VISIBLE_DEVICES=0,1 \
swift infer \
    --ckpt_dir "output/c4ai-command-r-plus/vx-xxx/checkpoint-xx" \
    --load_args_from_ckpt_dir true \
    --eval_human true \
    --temperature 0.3 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora false \
