CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_use_double_quant true \
    --output_dir Qwen2.5-1.5B-Instruct-BNB-NF4
