# v5 models / qwen3_next: inference for Qwen3-Next-80B-A3B (qwen3_next model_type).
#
# model_type = `qwen3_next` (resolved from --model). `--sampler vllm` serves the 80B-A3B sparse MoE with
# tensor + expert parallelism across the local GPUs; --vllm_enable_expert_parallel shards the experts.
# Interactive REPL by default (no --dataset); add --dataset + --result_path for an offline batch run, or
# --adapters output/vx-xxx/checkpoint-xxx to infer a fine-tune.
#
# REFERENCE-ONLY: not smoke-run here. Qwen3-Next-80B-A3B (80B total weights) needs multi-GPU VRAM even
# with 3B active; a single box cannot hold it in bf16. The flag shape mirrors ../../../infer/vllm/infer.sh.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
USE_SWIFT_V5=1 \
swift infer \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --sampler vllm \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --vllm_tensor_parallel_size 8 \
    --vllm_enable_expert_parallel true \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --stream true
