# v5 deploy: serve a generation model on the SGLang engine behind the OpenAI /chat/completions route.
#
# REFERENCE-ONLY: this machine has no SGLang kernel installed (`import sglang` -> ModuleNotFoundError), so
# this script is flag-validated against the v5 CLI but NOT smoke-run here. Install SGLang first
# (`pip install "sglang[all]"`) to run it. It is the exact counterpart of examples/deploy/vllm/deploy.sh --
# same twinkle-server (gateway app + sampler app on Ray Serve), same OpenAI surface, same client -- with the
# sampler engine switched to SGLang.
#
# --sampler sglang selects the SGLang sampler (dev maps it to twinkle's `sglang` / `sglang_async` sampler
# type; `sglang_async` when --enable_data_plane is on). Engine knobs take the `sglang_` prefix and live on
# RolloutConfig: --sglang_tp_size (tensor parallel), --sglang_mem_fraction_static (SGLang's analogue of
# vLLM's gpu_memory_utilization), --sglang_context_length, --sglang_dp_size/--sglang_ep_size for
# data/expert parallel. The vLLM-spelled --vllm_* knobs are rejected here.
#
# The gateway is mounted at --route_prefix (/api/v1), so clients hit {route_prefix}/chat/completions,
# /completions, /embeddings, /models, /infer, /health. --served_model_name is the `model` clients pass.
# route_prefix is /api/v1 (not /v1) so the token-in-token-out twinkle_client -- whose get_base_url() appends
# '/api/v1' -- resolves too; a plain OpenAI SDK client is unaffected.
#
# Generation knobs (temperature / max_tokens ...) are per-request, NOT deploy flags: the gateway applies each
# request's own sampling params, so a server-side default would be silently ignored. Only engine args belong
# here.
#
# Run the matching client with:  python examples/deploy/sglang/client.py
CUDA_VISIBLE_DEVICES=0,1 \
USE_SWIFT_V5=1 \
swift deploy \
    --model Qwen/Qwen3-8B \
    --sampler sglang \
    --served_model_name policy \
    --host 0.0.0.0 \
    --port 8000 \
    --route_prefix /api/v1 \
    --api_key EMPTY \
    --sglang_tp_size 2 \
    --sglang_mem_fraction_static 0.85 \
    --sglang_context_length 8192
