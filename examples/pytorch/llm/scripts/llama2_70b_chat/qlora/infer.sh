CUDA_VISIBLE_DEVICES=0,1 \
python src/llm_infer.py \
    --model_type llama2-70b-chat \
    --sft_type lora \
    --ckpt_dir "runs/llama2-70b-chat/vx_xxx/checkpoint-xxx" \
    --eval_human true \
    --quantization_bit 4 \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 20 \
    --top_p 0.9 \
    --do_sample true \
