MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters vx-xxx/checkpoint-xxx \
    --load_data_args true \
    --stream true \
    --max_new_tokens 2048
