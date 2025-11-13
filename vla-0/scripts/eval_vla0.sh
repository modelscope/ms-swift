PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0 \
IMAGE_MAX_TOKEN_NUM=1024 \
swift infer \
    --model /home/yuquan002/ssd/ms-swift-robotics/output/qwen3-vl-4b-instruct-vla0-libero/v5-20251029-163049/checkpoint-10000 \
    --stream true \
    --max_new_tokens 2048 \
    --load_data_args true