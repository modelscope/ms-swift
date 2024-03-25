# Experiment env: A10, RTX3090/4090, A100
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=1 \
python llm_infer.py \
    --ckpt_dir "/mnt/workspace/hujinghan.hjh/hjh/swift/examples/pytorch/llm/output/telechat-12b/v5-20240323-220717/checkpoint-3984" \
    --load_dataset_config true \
    --max_length 2048 \
    --use_flash_attn true \
    --max_new_tokens 2048 \
    --temperature 0.5 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora false \
    --dtype fp16 \
    --stream false