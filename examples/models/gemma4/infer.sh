# v5 models / gemma4: multimodal inference for Google Gemma-4 (gemma4 model_type).
#
# model_type = `gemma4` (resolved from --model). `--sampler vllm` serves generation through vLLM; use
# `--sampler transformers` for the HF engine. Interactive REPL by default (no --dataset): send an image or
# audio with the chat turn to exercise the multimodal path. To infer a fine-tuned adapter, replace --model
# with `--adapters output/vx-xxx/checkpoint-xxx` (base model, template and system come from args.json).
#
# REFERENCE-ONLY: not smoke-run here. google/gemma-4-* is GATED and not cached on this box; accept the
# license and download it first. The flag shape mirrors ../../../infer/vllm/infer.sh.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift infer \
    --model google/gemma-4-E2B-it \
    --sampler vllm \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --stream true
