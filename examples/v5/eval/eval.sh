# v5 eval: score a model on an EvalScope Native benchmark.
#
# `swift eval` builds a twinkle sampler in THIS process and hands it to EvalScope's Native runner -- there
# is no HTTP deployment and no remote service to point at. The model under test is the local sampler, so
# --sampler picks the engine (vllm / sglang / transformers) and any engine arg (e.g.
# --vllm_tensor_parallel_size, --vllm_gpu_memory_utilization) reaches it verbatim.
#
# EvalScope drives generation with --eval_num_proc concurrent requests. A continuous-batching backend
# (vllm/sglang) takes each trajectory as its own sample() call, so the engine stays saturated and a short
# answer never waits on the longest one in a coalesced batch.
#
# --eval_dataset names must be EvalScope Native benchmarks (e.g. gsm8k, mmlu, arc); they are normalized
# case-insensitively and an unknown name is rejected up front. --eval_limit caps samples per benchmark for
# a quick smoke run. Generation for the eval itself is set with --eval_generation_config (a JSON dict of
# EvalScope GenerateConfig fields), NOT the infer/deploy decoding flags.
#
# --eval_output_dir is where EvalScope writes reports; --result_jsonl appends one summary row per run.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift eval \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --sampler vllm \
    --eval_dataset gsm8k \
    --eval_limit 10 \
    --eval_num_proc 16 \
    --eval_generation_config '{"max_tokens": 1024, "temperature": 0.0}' \
    --vllm_gpu_memory_utilization 0.85 \
    --eval_output_dir ./eval_output \
    --result_jsonl ./eval_output/results.jsonl
