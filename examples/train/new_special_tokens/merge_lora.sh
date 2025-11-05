CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --merge_lora true


# infer
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx-merged \
    --max_batch_size 16 \
    --load_data_args true \
    --temperature 0
