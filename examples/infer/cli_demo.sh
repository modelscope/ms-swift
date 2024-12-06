CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend pt
