# Experimental environment: V100, A10, 3090
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --ckpt_dir "output/cogagent-18b-chat/vx-xxx/checkpoint-xx" \
    --load_args_from_ckpt_dir true \
    --eval_human true \
    --max_new_tokens 2048 \
    --temperature 0.3 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora false \
