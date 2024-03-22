# Experimental environment: A100
# 2 * 45GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
python llm_infer.py \
    --ckpt_dir "output/mixtral-moe-7b/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --use_flash_attn true \
    --max_new_tokens 2048 \
    --temperature 0.5 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora false \
