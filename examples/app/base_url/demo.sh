# You need to have a deployed model or api service first
CUDA_VISIBLE_DEVICES=0 swift app \
    --model '<model_name>' \
    --base_url http://127.0.0.1:8000/v1 \
    --stream true \
    --max_new_tokens 2048 \
    --lang zh
