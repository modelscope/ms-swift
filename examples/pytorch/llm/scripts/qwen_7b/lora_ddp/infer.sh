CUDA_VISIBLE_DEVICES=0 \
python src/llm_infer.py \
    --model_type qwen-7b \
    --sft_type lora \
    --template_type default-generation \
    --dtype bf16 \
    --ckpt_dir "runs/qwen-7b/vx_xxx/checkpoint-xxx" \
    --eval_human false \
    --dataset dureader-robust-zh \
    --train_dataset_sample -1 \
    --max_length 2048 \
    --use_flash_attn true \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 50 \
    --top_p 0.9 \
    --do_sample true \
