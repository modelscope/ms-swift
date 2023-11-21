# Experimental environment: A10
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --ckpt_dir "output/seqgpt-560m/vx_xxx/checkpoint-xxx" \
    --load_args_from_ckpt_dir true \
    --eval_human false \
    --max_length 1024 \
    --max_new_tokens 2048 \
    --temperature 0.3 \
    --repetition_penalty 1.05 \
    --do_sample true \
