# You need to use flash-attn (manual installation) instead of xformers.
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift infer \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --infer_backend vllm \
    --val_dataset speech_asr/speech_asr_aishell1_trainsets:validation#1000 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --max_new_tokens 2048 \
    --limit_mm_per_prompt '{"audio": 5}'
