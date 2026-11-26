# v5 interactive inference (TUI) on multiple GPUs.
#
# The default v5 CLI routes `swift infer` to the v5 dev CLI. With no --dataset the run opens the
# interactive REPL and `stream` defaults on. Multi-GPU here is vLLM tensor parallelism (TP=2), NOT
# data parallelism: the interactive guard (_guard_interactive) rejects dp>1 because every DP driver
# would open its own competing REPL, while a single driver sharded by TP inside one engine is fine.
# MEASURED: peak VRAM ~3 GiB/card (Qwen3.5-0.8B, batch smoke, 1 card(s), 2026-11-24)
CUDA_VISIBLE_DEVICES=0,1 \
swift infer \
    --model Qwen/Qwen3.5-4B \
    --sampler vllm \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.9 \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --stream true
