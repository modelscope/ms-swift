CUDA_VISIBLE_DEVICES=0 \
swift eval \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --eval_backend OpenCompass \
    --infer_backend sglang \
    --eval_limit 100 \
    --eval_dataset gsm8k
