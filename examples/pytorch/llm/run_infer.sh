CUDA_VISIBLE_DEVICES=0,1 \
python llm_infer.py \
    --model_type openbuddy-llama2-13b \
    --ckpt_dir "runs/openbuddy-llama2-13b/vx_xxx/checkpoint-xxx" \
    --eval_human true
