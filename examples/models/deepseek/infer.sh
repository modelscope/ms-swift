# v5 models / deepseek: inference for DeepSeek-V4 (deepseek_v4 model_type).
#
# model_type = `deepseek_v4` (resolved from --model). `--sampler vllm` serves the large MoE through vLLM
# with tensor parallelism across the local GPUs; drop to `--sampler transformers` for the HF engine. This
# launches the interactive REPL (no --dataset); add `--dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500'`
# and --result_path for an offline batch run. Point --adapters at a Megatron-trained checkpoint (converted
# back to HF with `swift export --to_hf`, see ../../../export/convert/) to infer a fine-tune.
#
# REFERENCE-ONLY: not smoke-run here. DeepSeek-V4-Flash is a multi-hundred-GB MoE; serving it needs a
# multi-GPU (typically multi-node) vLLM deployment with a large --vllm_tensor_parallel_size and enough
# aggregate VRAM. The flag shape mirrors the smoke-verified ../../../infer/vllm/infer.sh.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
USE_SWIFT_V5=1 \
swift infer \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --sampler vllm \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --vllm_tensor_parallel_size 8 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --stream true
