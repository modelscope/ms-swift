CUDA_VISIBLE_DEVICES=0,1 \
swift infer \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --infer_backend vllm \
    --val_dataset AI-ModelScope/LaTeX_OCR#1000 \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 2 \
    --max_model_len 32768 \
    --max_new_tokens 2048 \
    --limit_mm_per_prompt '{"image": 5, "video": 2}'
