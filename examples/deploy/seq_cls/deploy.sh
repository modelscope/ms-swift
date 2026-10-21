# v5 deploy: serve a text classifier behind the sampler's classify (pooling) forward.
#
# REFERENCE-ONLY. This is the single-text sibling of examples/deploy/reranker: same twinkle-server, same
# classify pooling head, same direct-to-sampler client path -- but the input is one text and the output is
# one logit per class instead of one relevance score per pair. It is marked reference-only because a
# sequence-classification deployment needs a *trained* classifier checkpoint (a base encoder such as
# ModernBERT-base has a randomly initialised head and produces meaningless logits), and no trained
# multi-class classifier is available in the local model cache to smoke against. Point --model at your own
# `swift sft --task_type seq_cls` output checkpoint and set --num_labels to match it.
#
# The classify client path itself IS verified end-to-end: examples/deploy/reranker/client.py drives the same
# sampler /twinkle/encode route with the same routing headers and gets correct, well-separated scores out of
# a real trained checkpoint (bge-reranker-base). This script differs only in --task_type seq_cls, --num_labels
# and the single-text client input.
#
# Key points, all shared with the reranker example:
#   * --task_type seq_cls selects the classify pooling head; a dedicated ...ForSequenceClassification
#     checkpoint is auto-detected by vLLM as a pooling model (no extra engine flag). A causal backbone reused
#     as a classifier instead needs `--vllm_engine_kwargs '{"runner":"pooling"}'`.
#   * classify has NO OpenAI gateway endpoint (the gateway's /embeddings hardcodes task='embed'), so the
#     client calls {route_prefix}/sampler/{served_model_name}/twinkle/encode directly with the Ray Serve
#     routing headers from twinkle_client.http.headers.build_routing_headers(served_name, api_key).
#   * --use_chat_template false makes twinkle tokenize a single user message's raw text, which is what a
#     plain classifier wants; leave it default for a model that ships its own classification chat template.
#   * --vllm_enable_prefix_caching false is REQUIRED for any pooling deploy: dev defaults it to true, but
#     vLLM 0.23's prefix-caching path asserts "HybridKVCacheCoordinator requires at least two attention
#     groups" on a single-group pooling model and the engine dies at startup.
#
# Run the matching client with:  python examples/deploy/seq_cls/client.py
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift deploy \
    --model /path/to/your/seq_cls-checkpoint \
    --task_type seq_cls \
    --num_labels 2 \
    --sampler vllm \
    --use_chat_template false \
    --served_model_name classifier \
    --host 0.0.0.0 \
    --port 8000 \
    --route_prefix /api/v1 \
    --api_key EMPTY \
    --vllm_enable_prefix_caching false \
    --vllm_enforce_eager true \
    --vllm_gpu_memory_utilization 0.9
