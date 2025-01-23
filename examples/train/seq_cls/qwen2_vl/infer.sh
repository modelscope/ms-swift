CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --load_data_args true
