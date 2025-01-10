CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --load_data_args true \
    --max_batch_size 16 \
    --truncation_strategy right \
    --max_length 512
