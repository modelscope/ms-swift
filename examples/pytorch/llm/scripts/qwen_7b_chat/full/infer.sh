# 19G
CUDA_VISIBLE_DEVICES=0 \
python src/llm_infer.py \
    --model_type qwen-7b-chat \
    --sft_type full \
    --template_type chatml \
    --dtype bf16 \
    --ckpt_dir "runs/qwen-7b-chat/vx_xxx/checkpoint-xxx" \
    --eval_human true \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 50 \
    --top_p 0.9 \
    --do_sample true \
