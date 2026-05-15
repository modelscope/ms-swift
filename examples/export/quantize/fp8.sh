# Due to the structural changes made to MoE architecture in `transformers>=5.0`,
# if you need to apply FP8 quantization to MoE models, please use `megatron export`
# (compatible with vLLM inference).
# Reference: https://github.com/modelscope/ms-swift/blob/main/examples/megatron/fp8/quant.sh
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen2.5-3B-Instruct \
    --quant_method fp8 \
    --output_dir Qwen2.5-3B-Instruct-FP8
