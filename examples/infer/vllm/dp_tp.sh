NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm \
    --val_dataset AI-ModelScope/alpaca-gpt4-data-zh#2000 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --vllm_tensor_parallel_size 2 \
    --max_new_tokens 2048 \
    --write_batch_size 1000
