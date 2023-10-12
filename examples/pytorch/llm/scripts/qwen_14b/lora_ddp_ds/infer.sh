# Experimental environment: A100
# If you want to merge LoRA weight and save it, you need to set `--merge_lora_and_save true`.
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python src/llm_infer.py \
    --model_type qwen-14b \
    --sft_type lora \
    --template_type default-generation \
    --dtype bf16 \
    --ckpt_dir "output/qwen-14b/vx_xxx/checkpoint-xxx" \
    --eval_human false \
    --dataset dureader-robust-zh \
    --max_length 2048 \
    --use_flash_attn true \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 20 \
    --top_p 0.9 \
    --do_sample true \
    --merge_lora_and_save false \
