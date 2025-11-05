# test_env: pip install "sglang[all]==0.4.6.*" -U
CUDA_VISIBLE_DEVICES=0 swift app \
    --model Qwen/Qwen2.5-7B-Instruct \
    --stream true \
    --infer_backend sglang \
    --max_new_tokens 2048 \
    --lang zh
