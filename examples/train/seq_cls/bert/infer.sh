CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model output/vx-xxx/checkpoint-xxx \
    --load_data_args true \
    --max_batch_size 16 \
    --truncation_strategy right
