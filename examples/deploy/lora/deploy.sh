# v5 deploy: serve a swift-trained LoRA checkpoint as an OpenAI endpoint.
#
# `swift deploy` reads the checkpoint's `args.json`, so `--adapters <path>` alone is enough: the base model,
# template and dtype are restored from the checkpoint and need not be repeated here.
#
# `--merge_lora true` folds the single adapter into the base weights on CPU before the server starts, then
# serves the merged model as ONE plain model under `--served_model_name`. There is no per-request adapter
# switching and no LoRA runtime overhead -- the client just talks to `merged` like any other model. This is
# the verified path (smoke-run end-to-end: merge -> serve -> chat completion).
#
# The gateway is mounted at --route_prefix (/api/v1); clients hit {route_prefix}/chat/completions with the
# OpenAI SDK. route_prefix is /api/v1 (not /v1) so the token-in-token-out twinkle_client resolves too.
#
# Run the matching client with:  python examples/deploy/lora/client.py
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift deploy \
    --adapters swift/test_lora \
    --merge_lora true \
    --sampler vllm \
    --served_model_name merged \
    --host 0.0.0.0 \
    --port 8000 \
    --route_prefix /api/v1 \
    --api_key EMPTY \
    --vllm_gpu_memory_utilization 0.9

# --- Online multi-LoRA hot-swap (REFERENCE-ONLY, not wired end-to-end yet) ------------------------------
# The intended one-server-many-adapters shape names each adapter so `/models` lists the base plus every
# adapter and a client picks one per request by passing that name as `model`:
#   swift deploy --adapters lora1=<path1> lora2=<path2> --sampler vllm --served_model_name base ...
# Naming works up to listing: `--adapters name=path` resolves (dev splits the served name off the path
# before the hub lookup) and `/models` returns ["base", "lora1", "lora2"]. But a chat request with
# model=lora1 currently 404s: the gateway resolves a request's model to the sampler mount via
# `_resolve_base_model`, which -- lacking any adapter->base metadata -- treats `lora1` as a directly
# supported base model (build_server_config puts adapter names in `supported_models`) and routes it to
# `/sampler/lora1`, while the sampler is only mounted at `/sampler/{served_model_name}`. Serving an adapter
# by name therefore needs the gateway to register each adapter's base_model and thread the adapter_name into
# the sampler request; until then, use `--merge_lora true` above (one deployment per adapter).
