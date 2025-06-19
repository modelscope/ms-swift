# test_env: pip install "sglang[all]==0.4.6.*" -U
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --infer_backend sglang \
    --stream true \
    --max_new_tokens 2048
