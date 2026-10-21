# v5 export: FP8 load-time quantization.
#
# Like bnb, fp8 is a LOAD-TIME scheme (compressed_tensors fits the scales as transformers materializes
# the weights), so dev loads the model with the fp8 quantization_config and saves a full, directly
# loadable directory -- mirroring legacy `swift export --quant_method fp8`. fp8 is the one method that
# does NOT require --quant_bits (the bit-width is implied by the scheme).
#
# Due to the structural changes made to the MoE architecture in transformers>=5.0, if you need FP8 on a
# MoE model, use `swift export --backend megatron` instead (compatible with vLLM inference). See
# examples/megatron/fp8/quant.sh. Needs compressed-tensors installed.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift export \
    --model Qwen/Qwen2.5-3B-Instruct \
    --quant_method fp8 \
    --output_dir Qwen2.5-3B-Instruct-FP8
