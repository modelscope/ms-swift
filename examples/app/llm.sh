CUDA_VISIBLE_DEVICES=0 swift app \
    --model Qwen/Qwen2.5-7B-Instruct \
    --stream true \
    --infer_backend pt \
    --max_new_tokens 2048 \
    --lang zh
