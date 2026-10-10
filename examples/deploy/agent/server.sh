CUDA_VISIBLE_DEVICES=0,1 swift deploy \
    --model Qwen/Qwen3.5-4B \
    --infer_backend vllm \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 2048 \
    --agent_template hermes \
    --served_model_name Qwen3.5-4B
