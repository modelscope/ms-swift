CUDA_VISIBLE_DEVICES=0 \
python src/llm_infer.py \
    --model_type polylm-13b \
    --sft_type lora \
    --dtype bf16 \
    --ckpt_dir "runs/polylm-13b/vx_xxx/checkpoint-xxx" \
    --eval_human true \
    --quantization_bit 4 \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 50 \
    --top_p 0.9 \
    --do_sample true \
