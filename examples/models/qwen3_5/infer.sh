# v5 models / qwen3_5: multimodal inference with the Qwen3.5 flagship (Qwen/Qwen3.5-4B).
#
# model_type = `qwen3_5` (resolved from --model). `--sampler vllm` serves generation through vLLM; use
# `--sampler transformers` for the plain HF engine (see ../../../infer/transformers/). This runs the
# interactive REPL by default (no --dataset): pass an image with the chat turn to exercise the vision
# path, or add `--dataset 'OK-VQA_train#64'` for an offline batch run writing --result_path.
#
# To infer a fine-tuned adapter instead of the base model, replace --model with
# `--adapters output/vx-xxx/checkpoint-xxx` (the base model, template and system are then read from the
# adapter's args.json). vLLM needs enough free VRAM on the selected GPU (gpu_memory_utilization default
# 0.9); lower it or pick an idle card if startup is refused.
#
# Requires transformers>=5.0.0.dev, qwen_vl_utils, decord. Reference for the vLLM engine flags:
# ../../../infer/vllm/infer.sh.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift infer \
    --model Qwen/Qwen3.5-4B \
    --sampler vllm \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --stream true
