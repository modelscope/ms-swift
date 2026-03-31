# use transformers==5.2.0
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen3.5-4B \
    --quant_method fp8 \
    --output_dir Qwen3.5-4B-FP8

# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --model Qwen3.5-4B-FP8
