# Test environment: transformers==4.47.1, autoawq==0.2.8
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
    --quant_n_samples 256 \
    --quant_batch_size -1 \
    --max_length 2048 \
    --quant_method awq \
    --quant_bits 4 \
    --output_dir Qwen2-VL-2B-Instruct-AWQ
