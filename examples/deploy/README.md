# v5 deploy examples

`swift deploy` serves a model as an OpenAI-compatible endpoint. It no longer hand-rolls a FastAPI app: it
builds a twinkle-server `ServerConfig` (a gateway app + a sampler app, plus a data-plane app for
token-in-token-out) and launches it on Ray Serve. With no cluster running, dev starts an in-process local
Ray head, so a single command serves. Every script sets `USE_SWIFT_V5=1` in its header.

Each server directory ships a `deploy.sh` (start the service) and a `client.py` (drive it):

| dir | what it serves | client |
| --- | --- | --- |
| `vllm/` | causal LM on vLLM: OpenAI chat + token-in-token-out data plane | `client.py`, `token_in_token_out_client.py` |
| `sglang/` | causal LM on the SGLang engine (reference-only: SGLang not installed here) | `client.py` |
| `embedding/` | an embedding model over OpenAI `/embeddings` | `client.py` |
| `reranker/` | a cross-encoder reranker (classify) — direct-to-sampler `/twinkle/encode` | `client.py` |
| `reward_model/` | a reward model (seq_cls, num_labels=1) — direct-to-sampler classify (reference-only) | `client.py` |
| `seq_cls/` | a sequence classifier over N labels — direct-to-sampler classify (reference-only) | `client.py` |
| `lora/` | a trained LoRA merged into its base and served as one model | `client.py` |
| `code/` | the programmatic entry: build Configs in Python and call `run_deploy` | `deploy.py` |

Notes that apply across the tree:

- The gateway is mounted at `--route_prefix`. Point an OpenAI SDK at `{route_prefix}` and pass
  `--served_model_name` as `model`. `route_prefix` is `/api/v1` (not `/v1`) in these examples so the
  token-in-token-out `twinkle_client` — which appends `/api/v1` to its base_url — resolves too.
- **Pooling tasks (embedding / reranker / seq_cls / reward) must pass `--vllm_enable_prefix_caching false`.**
  vLLM's hybrid KV-cache coordinator asserts on a single-attention-group pooling model when prefix caching
  is on, so the engine crashes at startup otherwise.
- **classify has no OpenAI endpoint.** The gateway's `/embeddings` hardcodes `task='embed'`, so a reranker /
  seq_cls / reward model is called by hitting the sampler directly at `{route_prefix}/sampler/{name}/twinkle/encode`
  with `pooling_params={'task': 'classify'}` and multiplex routing headers (see those `client.py`s). The
  sampler replica loads weights lazily, so a client must retry until it answers.
- Generation knobs (temperature / top_p / max_new_tokens …) are **per request**, not deploy flags — the
  gateway applies each request's own sampling params, so a server-side default is silently ignored.
