# You need to have a deployed model or api service first
swift eval \
  --model Your-model-name-here \
  --eval_url http://127.0.0.1:8000/v1/chat/completions \
  --eval_limit 10 \
  --eval_dataset gsm8k
