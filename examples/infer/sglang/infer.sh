# v5 batch inference on the SGLang engine (tensor-parallel).
#
# REFERENCE-ONLY: this machine has no SGLang kernel installed (`import sglang` -> ModuleNotFoundError), so
# this script is flag-validated against the v5 CLI but NOT smoke-run here. To run it, install SGLang first
# (`pip install "sglang[all]"`); the command below is otherwise identical in shape to ../vllm/infer.sh.
#
# USE_SWIFT_V5=1 routes `swift infer` to the v5 dev CLI. `--sampler sglang` picks the SGLang engine
# (`--infer_backend sglang` is the folded legacy alias). Engine knobs take the `sglang_` prefix and live on
# RolloutConfig: `--sglang_tp_size` (tensor parallel), `--sglang_mem_fraction_static` (SGLang's analogue of
# vLLM's gpu_memory_utilization), `--sglang_context_length`, `--sglang_dp_size`/`--sglang_ep_size` for
# data/expert parallel. The bare `--gpu_memory_utilization` spelling is a vLLM knob and is rejected here.
#
# A --dataset makes this an OFFLINE batch run (stream defaults off, one jsonl row per input at
# --result_path); drop --dataset for the interactive REPL. Generation args (--max_new_tokens/--temperature)
# are backend-agnostic and shared with the vllm/transformers examples.
#
# SGLang serves pooling tasks too: an embedding / seq_cls / reranker model runs through the engine's encode
# path -- but a decoder-only reranker (generative_reranker) must NOT be launched with SGLang's embedding
# flag; it scores via the yes/no generation path instead (see ../reranker/).
CUDA_VISIBLE_DEVICES=0,1 \
USE_SWIFT_V5=1 \
swift infer \
    --model Qwen/Qwen3-8B \
    --sampler sglang \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
    --sglang_tp_size 2 \
    --sglang_mem_fraction_static 0.85 \
    --sglang_context_length 8192 \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --output_format all \
    --result_path ./output/sglang_infer.jsonl
