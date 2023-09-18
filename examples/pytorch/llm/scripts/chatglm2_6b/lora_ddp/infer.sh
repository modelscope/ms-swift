CUDA_VISIBLE_DEVICES=0 \
python src/llm_infer.py \
    --model_type chatglm2-6b \
    --sft_type lora \
    --template_type chatglm2 \
    --dtype bf16 \
    --ckpt_dir "runs/chatglm2-6b/vx_xxx/checkpoint-xxx" \
    --eval_human false \
    --dataset code-python-zh \
    --max_length 8192 \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 20 \
    --top_p 0.9 \
    --do_sample true \
