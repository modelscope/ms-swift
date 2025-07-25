CUDA_VISIBLE_DEVICES=0,1 \
swift deploy \
    --model Qwen/Qwen3-8B \
    --infer_backend sglang \
    --max_new_tokens 2048 \
    --sglang_context_length 8192 \
    --sglang_tp_size 2 \
    --served_model_name Qwen3-8B

# After the server-side deployment above is successful, use the command below to perform a client call test.

# curl http://localhost:8000/v1/chat/completions \
# -H "Content-Type: application/json" \
# -d '{
# "model": "Qwen3-8B",
# "messages": [{"role": "user", "content": "What is your name?"}],
# "temperature": 0
# }'
