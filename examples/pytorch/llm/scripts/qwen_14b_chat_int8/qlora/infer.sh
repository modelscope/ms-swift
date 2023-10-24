# Experimental environment: V100, A10, 3090
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --model_type qwen-14b-chat-int8 \
    --sft_type lora \
    --template_type chatml \
    --dtype fp16 \
    --ckpt_dir "output/qwen-14b-chat-int8/vx_xxx/checkpoint-xxx" \
    --eval_human false \
    --dataset blossom-math-zh \
    --max_length 2048 \
    --use_flash_attn false \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 20 \
    --top_p 0.9 \
    --do_sample true \
    --merge_lora_and_save false \
