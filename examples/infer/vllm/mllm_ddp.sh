# You need to use flash-attn (manual installation) instead of xformers.
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift infer \
    --model Qwen/Qwen2.5-Omni-7B \
    --infer_backend vllm \
    --val_dataset speech_asr/speech_asr_aishell1_trainsets:validation#1000 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 2048 \
    --vllm_limit_mm_per_prompt '{"audio": 5}'
