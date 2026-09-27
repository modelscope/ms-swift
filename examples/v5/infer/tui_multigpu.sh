# v5 interactive inference (TUI) on multiple GPUs.
#
# USE_SWIFT_V5=1 routes `swift infer` to the v5 dev CLI. With no --dataset the run opens the
# interactive REPL and `stream` defaults on. Multi-GPU here is vLLM tensor parallelism (TP=2), NOT
# data parallelism: the interactive guard (_guard_interactive) rejects dp>1 because every DP driver
# would open its own competing REPL, while a single driver sharded by TP inside one engine is fine.
CUDA_VISIBLE_DEVICES=0,1 \
USE_SWIFT_V5=1 \
swift infer \
    --model Qwen/Qwen3.5-4B \
    --infer_backend vllm \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.9 \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --stream true
