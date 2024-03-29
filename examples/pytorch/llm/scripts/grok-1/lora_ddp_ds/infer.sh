# Experimental environment: 8 * A100
# Memory cost: 8 * 80G
PYTHONPATH=../../.. \
python llm_infer.py \
    --ckpt_dir output/grok-1/vxx-xxxx-xxxx/checkpoint-xxx \
    --dtype bf16 \
    --load_dataset_config true \
    --max_new_tokens 64 \
    --do_sample true \
    --dtype bf16 \
    --eval_human false \
    --merge_lora false \
