# v5 batch inference with the transformers (HF) backend over a DATASET.
#
# USE_SWIFT_V5=1 routes `swift infer` to the v5 dev CLI. `--sampler transformers` picks the plain
# HuggingFace forward/generate engine ('pt' is a compatibility alias of it); switch to `--sampler vllm`
# or `sglang` to run the same command on a served engine (see ../vllm/, ../sglang/). Because a --dataset
# is given this is an OFFLINE batch run, not the interactive REPL: `stream` defaults OFF and every row's
# completion is appended to --result_path (one jsonl row per input).
#
# The transformers backend is the one that runs a pooling task WITHOUT a pooling-capable engine: an
# embedding / seq_cls / reranker model forwards through HF here, whereas vLLM/SGLang serve those through
# the engine's own `encode` (see ../embedding/, ../reranker/, ../reward_model/). For plain causal_lm
# generation it batches --max_batch_size requests per engine call (vLLM/SGLang do their own continuous
# batching and ignore it).
#
# LoRA: point --adapters at one or more trained adapters (optionally name=path) instead of --model; the
# base model, template and system are read from the adapter's args.json, so you usually omit --model.
#
# HOW TO READ THIS RUN: each output row carries the input `messages` plus the generated `response`; with
# --metric acc/rouge the run also prints an aggregate score over the dataset at the end.
#
# This is a real inference scaffold. For a fast smoke run shrink the bounded dimensions only: a tiny slice
# (`#64`), a small --max_new_tokens, and a smaller model variant. Keep --sampler and --task_type -- they
# select the engine and the head.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --sampler transformers \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --max_batch_size 8 \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --output_format all \
    --result_path ./output/transformers_infer.jsonl
