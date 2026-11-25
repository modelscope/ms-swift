# v5 export: Quanto load-time quantization (INT4).
#
# quanto is a LOAD-TIME scheme like bnb/fp8/hqq/eetq: no calibration dataset, dev loads the model with the
# quanto quantization_config and saves a full, directly loadable directory. Accepted bit-widths for quanto
# are {2, 4, 8, float8} (validate.py); int4 is the common accuracy/size compromise. Needs `quanto` installed.
#
# REFERENCE-ONLY: quanto is not installed on this box, so this script was not smoke-run here. Install quanto,
# then run it as-is; a single GPU is enough for a 4B model.
# MEASURED: not run — quanto not installed (2026-11-24)
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen3.5-4B \
    --attn_impl flash_attention_2 \
    --quant_method quanto \
    --quant_bits 4 \
    --output_dir Qwen3.5-4B-Quanto-Int4
