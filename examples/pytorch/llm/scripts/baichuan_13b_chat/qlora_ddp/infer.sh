# 12G
CUDA_VISIBLE_DEVICES=0 \
python src/llm_infer.py \
    --model_type baichuan-13b-chat \
    --sft_type lora \
    --ckpt_dir "runs/baichuan-13b-chat/vx_xxx/checkpoint-xxx" \
    --eval_human true \
    --quantization_bit 4 \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 50 \
    --top_p 0.9 \
    --do_sample true \
