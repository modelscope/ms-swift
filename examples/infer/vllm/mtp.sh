CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --vllm_tensor_parallel_size 4 \
    --infer_backend vllm \
    --vllm_max_model_len 8192 \
    --val_dataset AI-ModelScope/alpaca-gpt4-data-zh#100 \
    --vllm_speculative_config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
    --vllm_gpu_memory_utilization 0.9 \
    --max_new_tokens 2048
