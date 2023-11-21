# Experimental environment: V100, A10, 3090
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --ckpt_dir "output/qwen-14b-chat-int4/vx_xxx/checkpoint-xxx" \
    --load_args_from_ckpt_dir true \
    --eval_human false \
    --max_length 4096 \
    --use_flash_attn false \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --do_sample true \
    --merge_lora_and_save false \
