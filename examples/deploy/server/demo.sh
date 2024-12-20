CUDA_VISIBLE_DEVICES=0 swift deploy \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm

# After the server-side deployment above is successful, use the command below to perform a client call test.

# curl http://localhost:8000/v1/chat/completions \
# -H "Content-Type: application/json" \
# -d '{
# "model": "Qwen2.5-7B-Instruct",
# "messages": [{"role": "user", "content": "晚上睡不着觉怎么办？"}],
# "temperature": 0
# }'
