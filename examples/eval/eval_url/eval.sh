# You need to have a deployed model or api service first
swift eval \
    --model '<model_name>' \
    --eval_backend OpenCompass \
    --eval_url http://127.0.0.1:8000/v1 \
    --eval_limit 100 \
    --eval_dataset gsm8k
