NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm \
    --val_dataset AI-ModelScope/alpaca-gpt4-data-zh#1000 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --max_new_tokens 2048
