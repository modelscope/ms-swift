# v5 eval: score a LoRA adapter, loaded live into the engine.
#
# --adapters points at a trained checkpoint; the engine is built with LoRA enabled and the adapter is
# selected for every request, so there is no merge step and no merged weights on disk. (eval scores a
# single model, so if several adapters are passed only the first is used.) The base model still comes from
# --model; --adapters layers the LoRA on top of it.
#
# Everything else matches eval.sh: an in-process twinkle sampler driven by EvalScope's Native runner, no
# deployment and no remote URL. To score the merged model instead, merge first (`swift export --merge_lora
# true`) and point --model at the merged checkpoint without --adapters.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift eval \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapters output/vx-xxx/checkpoint-xxx \
    --sampler vllm \
    --eval_dataset gsm8k \
    --eval_limit 10 \
    --eval_num_proc 16 \
    --eval_generation_config '{"max_tokens": 1024, "temperature": 0.0}' \
    --vllm_gpu_memory_utilization 0.85 \
    --eval_output_dir ./eval_output \
    --result_jsonl ./eval_output/results.jsonl
