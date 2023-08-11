# 14G
CUDA_VISIBLE_DEVICES=0 \
python src/llm_infer.py \
    --model_type chatglm2-6b \
    --sft_type lora \
    --ckpt_dir "runs/chatglm2-6b/vx_xxx/checkpoint-xxx" \
    --eval_human true \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 50 \
    --top_p 0.9 \
    --do_sample true \
