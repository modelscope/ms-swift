export USE_SWIFT_V5=0
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen2.5-Math-PRM-7B \
    --infer_backend transformers
