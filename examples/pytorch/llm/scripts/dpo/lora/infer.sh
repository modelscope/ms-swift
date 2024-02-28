# Experimental environment: A10, 3090
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --ckpt_dir output/mistral-7b/vx-xxx-xxx/checkpoint-xxx \
    --load_dataset_config true \
    --eval_human true \
    --use_flash_attn false \
    --max_new_tokens 1024 \
    --temperature 0.3 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora false \
