# v5 models / paddle_ocr: OCR inference with PaddleOCR-VL (PaddlePaddle/PaddleOCR-VL-1.6).
#
# model_type = `paddleocr_vl` (resolved from --model; template `paddle_ocr_1_5`). `--sampler vllm`
# serves generation through vLLM (PaddleOCRVLForConditionalGeneration is in vLLM's supported archs);
# use `--sampler transformers` for the plain HF engine. The ~1B model fits one card comfortably.
#
# Unlike the interactive families, OCR is naturally a batch task: this runs a slice of the handwriting
# LaTeX OCR corpus and writes each row's transcription to --result_path. Drop --dataset/--result_path
# to open the REPL instead (pass an image with the chat turn). To infer a fine-tuned adapter, replace
# --model with `--adapters output/vx-xxx/checkpoint-xxx` (base model, template and system are read
# from args.json).
#
# REFERENCE-ONLY: model_type verified against the loader registry; not smoke-run here (see train.sh).
# Requires transformers>=5.0. vLLM engine flag reference: ../../../infer/vllm/infer.sh.
# MEASURED (2026-11-24): not run as written. The vLLM spatial_merge_size AttributeError came from the
# transformers-native config this family declared then; the loader now trusts the checkpoint's remote
# code (transformers-engine infer is green since), but this vLLM command has not been re-run.
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model PaddlePaddle/PaddleOCR-VL-1.6 \
    --sampler vllm \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#32' \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 4096 \
    --max_new_tokens 1024 \
    --temperature 0 \
    --result_path ./output/paddle_ocr_infer.jsonl
