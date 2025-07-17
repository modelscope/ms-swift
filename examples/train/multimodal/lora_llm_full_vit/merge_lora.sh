CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --merge_lora true

# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --model output/vx-xxx/checkpoint-xxx-merged \
#     --stream true \
#     --load_data_args true \
#     --temperature 0 \
#     --max_new_tokens 2048
