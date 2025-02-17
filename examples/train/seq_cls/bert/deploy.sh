CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --adapters output/vx-xxx/checkpoint-xxx \
    --served_model_name bert-base-chinese \
    --truncation_strategy right \
    --max_length 512

# curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
# "model": "bert-base-chinese",
# "messages": [{"role": "user", "content": "包装差，容易被调包。"}]
# }'
