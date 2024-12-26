CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --adapters output/vx-xxx/checkpoint-xxx

# curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
# "model": "Qwen2.5-7B",
# "messages": [{"role": "user", "content": "Task: Sentiment Classification\nSentence: 包装差，容易被调包。\nCategory: negative, positive\nOutput:"}]
# }'
