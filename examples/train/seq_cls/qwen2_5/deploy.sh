CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --adapters output/vx-xxx/checkpoint-xxx

# curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
# "model": "Qwen2.5-0.5B",
# "messages": [{"role": "user", "content": "包装差，容易被调包。"}]
# }'
