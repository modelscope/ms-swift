# Experimental environment: A10
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --ckpt_dir "output/internlm2-7b-sft-chat/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --max_new_tokens 2048 \
    --temperature 0.5 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --stream false \
    --do_sample true \
    --merge_lora false \
