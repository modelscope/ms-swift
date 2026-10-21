# v5 deploy: serve a reward model (scalar score for a whole conversation) via the classify pooling forward.
#
# REFERENCE-ONLY. A reward model is architecturally a `seq_cls` task with `num_labels=1` built on a *causal*
# backbone: swift derives exactly that pairing (see swift/dev/config/process.py:687). Two consequences shape
# this script, and one of them blocks a local smoke:
#
#   * Because the backbone is causal, vLLM would read the checkpoint as a generation model, so the pooling
#     runner must be pinned explicitly with `--vllm_engine_kwargs '{"runner":"pooling"}'` (a dedicated
#     ...ForSequenceClassification checkpoint such as the reranker's is auto-detected and needs no such flag).
#   * --vllm_enable_prefix_caching false is REQUIRED for any pooling deploy: dev defaults it to true, but
#     vLLM 0.23's prefix-caching path asserts "HybridKVCacheCoordinator requires at least two attention
#     groups" on a single-group pooling model and the engine dies at startup.
#   * Public reward checkpoints (internlm2-1_8b-reward here) ship custom modelling code and need
#     `--model_kwargs '{"trust_remote_code":true}'`. Deploying one currently FAILS: `swift deploy` pins the
#     chat template inside the Ray Serve sampler actor at construction, and the remote-code loader sets a
#     `signal.alarm` download timeout that raises "signal only works in main thread of the main interpreter"
#     in that worker thread (then the loader rejects the custom code outright). This is a twinkle-loader
#     kernel limitation, not something an example can route around, so the script is reference-only. A reward
#     checkpoint that needs no remote code would deploy fine.
#
# The classify client path itself is verified end-to-end by examples/deploy/reranker/client.py (same sampler
# /twinkle/encode route, same routing headers, correct scores from a real trained checkpoint); this example
# differs only in the causal-backbone runner flag and in sending a full conversation instead of a pair.
#
# Run the matching client with:  python examples/deploy/reward_model/client.py
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift deploy \
    --model Shanghai_AI_Laboratory/internlm2-1_8b-reward \
    --task_type seq_cls \
    --num_labels 1 \
    --sampler vllm \
    --model_kwargs '{"trust_remote_code": true}' \
    --vllm_engine_kwargs '{"runner": "pooling"}' \
    --served_model_name reward \
    --host 0.0.0.0 \
    --port 8000 \
    --route_prefix /api/v1 \
    --api_key EMPTY \
    --vllm_enable_prefix_caching false \
    --vllm_enforce_eager true \
    --vllm_gpu_memory_utilization 0.9
