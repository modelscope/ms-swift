CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --model output/vx-xxx/checkpoint-xxx \
    --served_model_name bert-base-chinese

# curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
# "model": "bert-base-chinese",
# "messages": [{"role": "user", "content": "Task: Sentiment Classification\nSentence: 包装差，容易被调包。\nCategory: negative, positive\nOutput:"}]
# }'
