# v5 models / gpt_oss: inference for OpenAI gpt-oss-20b (gpt_oss model_type).
#
# model_type = `gpt_oss` (resolved from --model). `--sampler vllm` serves the 20B MoE through vLLM; use
# `--sampler transformers` for the HF engine. Interactive REPL by default (no --dataset); add --dataset +
# --result_path for an offline batch run, or --adapters output/vx-xxx/checkpoint-xxx to infer a fine-tune.
# gpt-oss uses a harmony-style chat format with reasoning; the template is resolved from the checkpoint.
#
# REFERENCE-ONLY: not smoke-run here. openai-mirror/gpt-oss-20b is not cached on this box (~42GB). The
# flag shape mirrors the smoke-verified ../../../infer/vllm/infer.sh.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift infer \
    --model openai-mirror/gpt-oss-20b \
    --sampler vllm \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --stream true
