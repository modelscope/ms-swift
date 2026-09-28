# v5 deploy: serve a model as an OpenAI endpoint AND a token-in-token-out sampler.
#
# `swift deploy` no longer hand-rolls a FastAPI app; it builds a twinkle-server ServerConfig (a gateway
# app + a sampler app) and launches it on Ray Serve. With no cluster running, dev starts an in-process
# local Ray head, so this single command serves -- pass `--mode ray` (or `--ray_address auto`) instead to
# attach to a cluster that already exists.
#
# The gateway is mounted at --route_prefix, so clients hit /v1/chat/completions, /v1/completions,
# /v1/embeddings, /v1/models, /v1/infer and /health. --served_model_name is the `model` clients pass and
# the name the sampler is mounted under ({route_prefix}/sampler/{served_model_name}).
#
# --enable_data_plane is what turns on token-in-token-out for an RL rollout: it adds a data-plane app at
# {route_prefix}/data-plane and switches the sampler to vllm_async (the only backend that can hand a
# rollout a server-side DataRef). Drop it for the lightweight path, where the sampler returns each
# sequence's trainable features (new_input_feature) inline and no data plane is needed.
#
# Run the matching client with:  python examples/v5/deploy/token_in_token_out_client.py
#
# Note: generation knobs (temperature / top_p / max_new_tokens ...) are NOT deploy flags -- the gateway
# applies each request's own sampling params, so a server-side default would be silently ignored (dev
# warns if you set one). Send them per request instead; only engine args like --vllm_gpu_memory_utilization
# belong here.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift deploy \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --infer_backend vllm \
    --served_model_name policy \
    --host 0.0.0.0 \
    --port 8000 \
    --route_prefix /v1 \
    --api_key EMPTY \
    --enable_data_plane true \
    --vllm_gpu_memory_utilization 0.85
