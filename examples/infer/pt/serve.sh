CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --model_id_or_path qwen/Qwen2-7B-Instruct \
    --infer_backend pt
