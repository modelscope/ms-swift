CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --load_args false \
    --model output/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
