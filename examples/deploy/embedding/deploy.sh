# v5 deploy: serve an embedding model with vLLM's pooling runner behind a single gateway.
#
# `swift deploy` under USE_SWIFT_V5=1 routes to the dev CLI (swift.dev.cli.deploy), which builds a
# twinkle-server ServerConfig -- a gateway app plus sampler apps on Ray Serve, no hand-rolled
# FastAPI. --task_type embedding selects the embedding (pooling) forward: the server exposes the
# OpenAI /embeddings surface; vLLM auto-detects the pooling runner from the checkpoint. The 300M
# model keeps the example light; swap in a larger embedding model for a real deployment.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift deploy \
    --model google/embeddinggemma-300m \
    --task_type embedding \
    --sampler vllm \
    --use_chat_template false \
    --served_model_name embedder \
    --host 0.0.0.0 \
    --port 8000 \
    --route_prefix /api/v1 \
    --api_key EMPTY \
    --vllm_max_model_len 2048 \
    --vllm_enable_prefix_caching false \
    --vllm_enforce_eager true \
    --vllm_gpu_memory_utilization 0.9
