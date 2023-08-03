CUDA_VISIBLE_DEVICES=0,1 \
python llm_infer.py \
    --model_type qwen-7b \
    --ckpt_dir "qwen-7b/vx_xxx/checkpoint-xxx" \
    --eval_human true \
    --dataset_sample 20000
