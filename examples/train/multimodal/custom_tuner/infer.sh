CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --load_args false \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --load_data_args true \
    --temperature 0 \
    --max_new_tokens 2048
