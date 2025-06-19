CUDA_VISIBLE_DEVICES=0,1 \
swift infer \
    --model Qwen/Qwen3-8B \
    --infer_backend sglang \
    --stream true \
    --max_new_tokens 2048 \
    --sglang_context_length 8192 \
    --sglang_tp_size 2
