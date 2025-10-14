CUDA_VISIBLE_DEVICES=0,1 swift deploy \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --infer_backend vllm \
    --served_model_name Qwen2.5-VL-7B-Instruct \
    --vllm_max_model_len 8192 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_data_parallel_size 2

# After the server-side deployment above is successful, use the command below to perform a client call test.

# curl http://localhost:8000/v1/chat/completions \
# -H "Content-Type: application/json" \
# -d '{
# "model": "Qwen2.5-VL-7B-Instruct",
# "messages": [{"role": "user", "content": [
#     {"type": "image", "image": "http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png"},
#     {"type": "image", "image": "http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png"},
#     {"type": "text", "text": "What is the difference between the two images?"}
# ]}],
# "max_tokens": 256,
# "temperature": 0
# }'
