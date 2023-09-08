CUDA_VISIBLE_DEVICES=0 \
python src/llm_infer.py \
    --model_type qwen-vl \
    --sft_type lora \
    --template_type default \
    --dtype bf16 \
    --ckpt_dir "runs/qwen-vl/vx_xxx/checkpoint-xxx" \
    --eval_human false \
    --dataset coco-en \
    --dataset_sample 20000 \
    --use_flash_attn true \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 50 \
    --top_p 0.9 \
    --do_sample true \
