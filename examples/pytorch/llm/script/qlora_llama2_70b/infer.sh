# 40G
CUDA_VISIBLE_DEVICES=0,1 \
python llm_infer.py \
    --model_type llama2-7b \
    --sft_type lora \
    --ckpt_dir "runs/llama2-70b/vx_xxx/checkpoint-xxx" \
    --eval_human true \
    --quantization_bit 4 \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 50 \
    --top_p 0.9 \
    --do_sample true \
