CUDA_VISIBLE_DEVICES=0,1 \
python src/llm_infer.py \
    --model_type openbuddy-llama2-70b \
    --sft_type lora \
    --template_type openbuddy-llama \
    --dtype bf16 \
    --ckpt_dir "runs/openbuddy-llama2-70b/vx_xxx/checkpoint-xxx" \
    --eval_human true \
    --quantization_bit 4 \
    --bnb_4bit_comp_dtype bf16 \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 20 \
    --top_p 0.9 \
    --do_sample true \
