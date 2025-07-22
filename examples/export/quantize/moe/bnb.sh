CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen3-30B-A3B \
    --quant_method bnb \
    --quant_bits 4 \
    --output_dir Qwen3-30B-A3B-BNB-Int4
