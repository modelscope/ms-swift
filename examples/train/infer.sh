# If it's full parameter training, use `--model xxx` instead of `--adapters xxx`.
# If you are using the validation set for inference, add the parameter `--load_data_args true`.
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
