CUDA_VISIBLE_DEVICES=0 \
swift eval \
    --ckpt_dir "output/qwen-7b-chat/vx-xxx/checkpoint-xxx" \
    --eval_dataset arc \
    --eval_limit 10 \
    --infer_backend pt \
    --custom_eval_config eval_example/custom_config.json
