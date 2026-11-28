# v5 deploy: serve a cross-encoder reranker behind the sampler's classify (pooling) forward.
#
# A reranker is a sequence-classification model with a single output logit: it scores a (query, passage)
# pair, not a lone string. `swift deploy` builds the same twinkle-server as examples/v5/deploy/embedding (a
# gateway app + a sampler app on Ray Serve); --task_type reranker selects the classify pooling head and vLLM
# auto-detects the pooling runner from the checkpoint (its log prints "Resolved `--runner auto` to
# `--runner pooling`"), so no extra engine flag is needed for a dedicated reranker.
#
# THE CLIENT PATH DIFFERS FROM EMBEDDING. The gateway's OpenAI surface only exposes /embeddings, and that
# route hardcodes the pooling task to 'embed' (twinkle_server/gateway/openai_bridge.py). There is NO OpenAI
# endpoint for classify, so a reranker/seq_cls/reward client cannot use the OpenAI SDK. Instead it calls the
# sampler's own /twinkle/encode route directly with pooling_params {'task': 'classify'}. Each served model
# gets its own single-copy sampler mounted at {route_prefix}/sampler/{served_model_name} (the old Ray Serve
# multiplexed sticky routing is gone), so the request only needs the identity/auth headers the gateway
# would otherwise inject -- twinkle.protocol.headers.build_request_headers(request_id, 'Bearer ' + api_key)
# builds them (x-request-id + Authorization/Twinkle-Authorization). See client.py.
#
# --use_chat_template false: a cross-encoder has no chat template. With the chat template off, twinkle's
# template tokenizes a single user message's raw text, so the client packs the pair into one message using
# the model's own separator (for the XLM-RoBERTa-based bge-reranker, "</s></s>" between query and passage
# reproduces the exact <s>q</s></s>p</s> pair encoding). A reranker that ships a query/passage chat template
# should instead leave this at its default.
#
# --vllm_max_model_len is pinned to the checkpoint's ceiling (bge-reranker-base is 512).
#
# --vllm_enable_prefix_caching false is REQUIRED for a pooling deploy. dev defaults it to true, but vLLM
# 0.23's prefix-caching path builds a hybrid KV-cache coordinator that asserts "requires at least two
# attention groups" on a single-group pooling/encoder model, so the engine dies at startup. A pooling model
# runs one forward and reuses no KV across steps, so prefix caching buys nothing here anyway.
#
# Run the matching client with:  python examples/v5/deploy/reranker/client.py
# MEASURED: peak VRAM ~2 GiB/card (bge-reranker-base, server smoke, 1 card(s), 2026-11-24)
CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --model BAAI/bge-reranker-base \
    --task_type reranker \
    --sampler vllm \
    --use_chat_template false \
    --served_model_name reranker \
    --host 0.0.0.0 \
    --port 8000 \
    --route_prefix /api/v1 \
    --api_key EMPTY \
    --vllm_max_model_len 512 \
    --vllm_enable_prefix_caching false \
    --vllm_enforce_eager true \
    --vllm_gpu_memory_utilization 0.9
