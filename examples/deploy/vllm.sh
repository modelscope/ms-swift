CUDA_VISIBLE_DEVICES=0 swift deploy \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm \
    --served_model_name Qwen2.5-7B-Instruct

# After the server-side deployment above is successful, use the command below to perform a client call test.

# curl http://localhost:8000/v1/chat/completions \
# -H "Content-Type: application/json" \
# -d '{
# "model": "Qwen2.5-7B-Instruct",
# "messages": [{"role": "user", "content": "What is your name?"}],
# "temperature": 0
# }'
