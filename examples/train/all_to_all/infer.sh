# 53GiB
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model BAAI/Emu3-Gen \
    --infer_backend pt \
    --stream False \
    --use_chat_template False \
    --top_k 2048 \
    --max_new_tokens 40960
