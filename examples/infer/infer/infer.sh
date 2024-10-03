CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model_id_or_path LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --infer_backend pt \
    --eval_human true
