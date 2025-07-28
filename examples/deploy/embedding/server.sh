# GME/GTE models or your checkpoints are also supported
# pt/vllm/sglang supported
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --host 0.0.0.0 \
    --port 8000 \
    --model Qwen/Qwen3-Embedding-0.6B \
    --infer_backend sglang
