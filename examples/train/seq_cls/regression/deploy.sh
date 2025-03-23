CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --adapters output/vx-xxx/checkpoint-xxx

# curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
# "model": "Qwen2.5-0.5B",
# "messages": [{"role": "user", "content": "Task: Based on the given two sentences, provide a similarity score between 0.0 and 1.0.\nSentence 1: The animal is eating.\nSentence 2: A woman is dancing.\nSimilarity score: "}]
# }'
