# v5 export: AWQ calibration-based quantization (4-bit).
#
# awq is a CALIBRATION method (unlike bnb/fp8): dev loads the model in full precision, builds calibration
# batches from --dataset, fits the activation-aware scales, packs the weights and saves a full loadable
# directory. The MoE expert grouping (modules_in_block_to_quantize) and the lm_head exclusion are handled
# inside twinkle's AwqQuantizer, so no extra flags are needed for them here.
#
# --quant_n_samples calibration rows, --quant_batch_size rows per forward, --max_length the truncation
# length, --group_size weights per scale. --device_map cpu keeps a large model off the GPU during the
# (slow) calibration pass, as legacy does. Requires autoawq installed.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift export \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
    --device_map cpu \
    --quant_n_samples 256 \
    --quant_batch_size 1 \
    --max_length 2048 \
    --quant_method awq \
    --quant_bits 4 \
    --output_dir Qwen2.5-7B-Instruct-AWQ
