CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift app \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --stream true \
    --infer_backend vllm \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --max_new_tokens 2048 \
    --limit_mm_per_prompt '{"image": 5, "video": 2}' \
    --lang zh
