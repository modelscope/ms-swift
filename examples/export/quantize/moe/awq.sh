CUDA_VISIBLE_DEVICES=0,1 \
swift export \
    --model Qwen/Qwen3-30B-A3B \
    --dataset 'swift/Qwen3-SFT-Mixin' \
    --device_map auto \
    --quant_n_samples 64 \
    --quant_batch_size -1 \
    --max_length 8192 \
    --quant_method awq \
    --quant_bits 4 \
    --output_dir Qwen3-30B-A3B-AWQ
