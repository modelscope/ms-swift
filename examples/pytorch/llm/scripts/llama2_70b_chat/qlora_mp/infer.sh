# Experimental environment: 2 * 3090
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
python llm_infer.py \
    --ckpt_dir "output/llama2-70b-chat/vx_xxx/checkpoint-xxx" \
    --load_args_from_ckpt_dir true \
    --eval_human false \
    --max_length 2048 \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --do_sample true \
    --merge_lora_and_save false \
