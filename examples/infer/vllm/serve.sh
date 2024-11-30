CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --model Qwen/Qwen2-7B-Instruct \
    --infer_backend vllm
