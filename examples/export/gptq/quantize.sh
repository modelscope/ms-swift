# v5 export: GPTQ calibration-based quantization (4-bit).
#
# gptq is a CALIBRATION method: dev loads the model, fits per-group scales on --dataset, packs to the
# GPTQ checkpoint format and saves a full loadable directory. twinkle's GptqQuantizer owns the details
# that legacy hand-rolled -- the MoE expert grouping (modules_in_block_to_quantize), the gptq_v2 dynamic
# tied-weights keys, and the optimum checkpoint_format/format version difference -- so the CLI surface is
# just the method + calibration knobs.
#
# gptq_v2 is the same command with `--quant_method gptq_v2` (it needs gptqmodel installed); plain gptq
# needs auto-gptq/gptqmodel. OMP_NUM_THREADS=14 works around a known AutoGPTQ threading issue
# (https://github.com/AutoGPTQ/AutoGPTQ/issues/439).
OMP_NUM_THREADS=14 \
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift export \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
    --quant_n_samples 256 \
    --quant_batch_size 1 \
    --max_length 2048 \
    --quant_method gptq \
    --quant_bits 4 \
    --output_dir Qwen2.5-1.5B-Instruct-GPTQ-Int4
