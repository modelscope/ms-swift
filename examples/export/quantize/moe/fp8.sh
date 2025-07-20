CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen3-30B-A3B \
    --quant_method fp8 \
    --output_dir Qwen3-30B-A3B-FP8

# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --model Qwen3-30B-A3B-FP8 \
#     --infer_backend vllm \
#     --stream true
