CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen2.5-72B-Instruct \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
    --device_map cpu \
    --quant_n_samples 256 \
    --quant_batch_size 1 \
    --max_length 2048 \
    --quant_method awq \
    --quant_bits 4 \
    --output_dir Qwen2.5-72B-Instruct-AWQ
