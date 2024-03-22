# Experiment env: A10, RTX3090/4090, A100
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
python llm_infer.py \
    --ckpt_dir "output/llama2-7b-aqlm-2bit-1x16/vx-xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --use_flash_attn true \
    --max_new_tokens 2048 \
    --temperature 0.5 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --stream false \
    --merge_lora false \
