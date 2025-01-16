CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen2.5-Math-PRM-7B \
    --infer_backend pt
