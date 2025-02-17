# gptq quantize
CUDA_VISIBLE_DEVICES=0 swift export \
    --model Shanghai_AI_Laboratory/internlm2-1_8b-reward \
    --output_dir output/internlm2-1_8b-reward-gptq-int4 \
    --quant_bits 4 \
    --quant_method gptq \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#1000' 'AI-ModelScope/alpaca-gpt4-data-en#1000'

# infer
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model output/internlm2-1_8b-reward-gptq-int4 \
    --val_dataset 'AI-ModelScope/alpaca-gpt4-data-zh#1000' \
    --max_batch_size 16
