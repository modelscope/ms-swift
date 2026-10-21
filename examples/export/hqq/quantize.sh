# v5 export: HQQ load-time quantization (4-bit).
#
# hqq (Half-Quadratic Quantization) is a LOAD-TIME scheme like bnb/fp8: it quantizes from the weights
# alone (no calibration dataset), so dev loads the model with the hqq quantization_config and saves a full,
# directly loadable directory. Two hqq-specific knobs are surfaced on QuantizeConfig:
#   --hqq_axis      the axis the scales are computed along (None lets HQQ pick its default);
#   --group_size    weights sharing one scale (smaller = more accurate, larger on disk; must divide the
#                   layer's input dim).
# Accepted bit-widths for hqq are {1, 2, 3, 4, 8}. Needs `hqq` installed.
#
# REFERENCE-ONLY: hqq is not installed on this box, so this script was not smoke-run here. Install hqq, then
# run it as-is; a single GPU is enough for a 1.5B model.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift export \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --quant_method hqq \
    --quant_bits 4 \
    --group_size 64 \
    --output_dir Qwen2.5-1.5B-Instruct-HQQ-Int4
