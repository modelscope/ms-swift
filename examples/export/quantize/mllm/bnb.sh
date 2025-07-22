CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --quant_method bnb \
    --quant_bits 4 \
    --output_dir Qwen2.5-VL-3B-Instruct-BNB-Int4
