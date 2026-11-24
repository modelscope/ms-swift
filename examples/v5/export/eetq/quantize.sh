# v5 export: EETQ load-time quantization (INT8).
#
# eetq is a LOAD-TIME scheme like bnb/fp8/hqq: no calibration dataset, dev loads the model with the eetq
# quantization_config and saves a full, directly loadable directory. EETQ only supports INT8, so
# --quant_bits is fixed at 8 (validate.py rejects any other width for eetq). Needs `eetq` installed.
#
# REFERENCE-ONLY: eetq is not installed on this box, so this script was not smoke-run here. Install eetq,
# then run it as-is; a single GPU is enough for a 4B model.
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen3.5-4B \
    --attn_impl flash_attention_2 \
    --quant_method eetq \
    --quant_bits 8 \
    --output_dir Qwen3.5-4B-EETQ-Int8
