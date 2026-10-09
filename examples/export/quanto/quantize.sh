# v5 export: Quanto load-time quantization (INT4).
#
# quanto is a LOAD-TIME scheme like bnb/fp8/hqq/eetq: no calibration dataset, dev loads the model with the
# quanto quantization_config and saves a full, directly loadable directory. Accepted bit-widths for quanto
# are {2, 4, 8, float8} (validate.py); int4 is the common accuracy/size compromise. Needs `quanto` installed.
#
# REFERENCE-ONLY: quanto is not installed on this box, so this script was not smoke-run here. Install quanto,
# then run it as-is; a single GPU is enough for a 1.5B model.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift export \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --quant_method quanto \
    --quant_bits 4 \
    --output_dir Qwen2.5-1.5B-Instruct-Quanto-Int4
