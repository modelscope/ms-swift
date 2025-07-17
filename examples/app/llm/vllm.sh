CUDA_VISIBLE_DEVICES=0 swift app \
    --model Qwen/Qwen2.5-7B-Instruct \
    --stream true \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --lang zh
