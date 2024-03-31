# Experimental environment: 3090
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --ckpt_dir "output/codegeex2-6b/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora false \
