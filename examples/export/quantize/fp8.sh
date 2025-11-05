CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen2.5-3B-Instruct \
    --quant_method fp8 \
    --output_dir Qwen2.5-3B-Instruct-FP8
