CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --max_new_tokens 2048 \
    --load_data_args true \
    --max_batch_size 16
