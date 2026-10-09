# v5 deploy: serve an embedding model behind the OpenAI /embeddings endpoint.
#
# This is the pooling counterpart of examples/deploy/vllm (which serves generation). `swift deploy` builds
# the same twinkle-server (a gateway app + a sampler app on Ray Serve); the difference is the model runs a
# pooling head instead of an LM head, and clients call {route_prefix}/embeddings rather than
# {route_prefix}/chat/completions.
#
# Two things make an embedding deployment work, and both are easy to get wrong:
#
# 1. --task_type embedding selects the pooling forward. vLLM then auto-detects the pooling runner from the
#    checkpoint (its log prints "Resolved `--runner auto` to `--runner pooling`"), so the LM head is swapped
#    for the pooling head and the sampler serves requests through `encode`. A dedicated embedding checkpoint
#    needs no extra engine flag; only a model that vLLM would otherwise read as a generation model (e.g. a
#    causal backbone reused for classify) needs an explicit `--vllm_engine_kwargs '{"runner":"pooling"}'`.
#
# 2. --use_chat_template false. The gateway's /embeddings wraps each input string as a single-user-turn
#    trajectory and the served sampler encodes it with twinkle's template. An embedding checkpoint such as
#    embeddinggemma ships no tokenizer.chat_template, so the default chat-template path raises
#    "Cannot use chat template functions because tokenizer.chat_template is not set". Turning the chat
#    template off makes the template tokenize the raw text directly (it asserts a single user message, which
#    is exactly what /embeddings sends). A model that DOES define a query/passage chat template
#    (e.g. Qwen3-Embedding) should instead leave this at its default and rely on that template.
#
# --vllm_max_model_len is pinned to the checkpoint's own ceiling (embeddinggemma's max_position_embeddings
# is 2048); asking for more makes vLLM reject the config at startup.
#
# --vllm_enable_prefix_caching false is REQUIRED for a pooling deploy. dev defaults it to true, but vLLM
# 0.23's prefix-caching path builds a hybrid KV-cache coordinator that asserts "requires at least two
# attention groups" on a single-group pooling model, so the engine dies at startup. A pooling model runs one
# forward and reuses no KV across steps, so prefix caching buys nothing here anyway.
#
# Run the matching client with:  python examples/deploy/embedding/client.py
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
