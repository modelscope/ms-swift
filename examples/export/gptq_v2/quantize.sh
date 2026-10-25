# v5 export: GPTQ-v2 calibration-based quantization (4-bit).
#
# gptq_v2 is the newer GPTQ checkpoint format: same calibration flow as ../gptq/ (dev loads the model,
# fits per-group scales on --dataset, packs and saves a full loadable directory), but it writes the v2
# format with the dynamic tied-weights keys that twinkle's GptqQuantizer handles internally -- so the CLI
# surface is identical to gptq apart from `--quant_method gptq_v2`. It needs `gptqmodel` installed
# (plain gptq accepts auto-gptq OR gptqmodel). OMP_NUM_THREADS=14 works around a known AutoGPTQ threading
# issue (https://github.com/AutoGPTQ/AutoGPTQ/issues/439).
#
# REFERENCE-ONLY: gptqmodel is not installed on this box, so this script was not smoke-run here. Install
# gptqmodel, then run it as-is (a single GPU is enough for a 1.5B model; --device_map is retired -- to
# keep calibration off the GPU at the cost of speed, run with CUDA_VISIBLE_DEVICES='').
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
    --quant_method gptq_v2 \
    --quant_bits 4 \
    --output_dir Qwen2.5-1.5B-Instruct-GPTQ-V2-Int4
