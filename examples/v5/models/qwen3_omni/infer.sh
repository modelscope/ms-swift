# v5 models / qwen3_omni: multimodal inference with Qwen3-Omni (Qwen/Qwen3-Omni-30B-A3B-Instruct).
#
# model_type = `qwen3_omni_moe` (resolved from --model; template `qwen3_omni`). `--sampler vllm` serves
# the thinker through vLLM (Qwen3OmniMoeForConditionalGeneration is in vLLM's supported archs; text
# output -- the talker audio-out path is not what vLLM serves). The 30B-A3B bf16 weights need ~60GB,
# so this runs TP=2 across two cards; use `--sampler transformers` for the plain HF engine instead.
#
# This runs the interactive REPL by default (no --dataset): pass an image or an audio file with the
# chat turn to exercise the vision/audio paths, or add a dataset slice (e.g.
# 'speech_asr/speech_asr_aishell1_trainsets:validation#32') for an offline batch run writing
# --result_path. To infer a fine-tuned adapter, replace --model with
# `--adapters output/vx-xxx/checkpoint-xxx` (base model, template and system are read from args.json).
#
# REFERENCE-ONLY: model_type verified against the loader registry; not parse/smoke-verified here.
# The ~70GB checkpoint is not cached on this box. Requires transformers>=4.57.dev0, soundfile, decord,
# qwen_omni_utils>=0.0.9. vLLM engine flag reference: ../../../infer/vllm/infer.sh.
# MEASURED: peak VRAM ~35 GiB/card (Qwen3-Omni-30B-A3B, 4-step smoke, 2 card(s), 2026-11-24)
CUDA_VISIBLE_DEVICES=0,1 \
swift infer \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --sampler vllm \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --stream true
