# sh examples/custom/infer.sh
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --ckpt_dir output/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
    --infer_backend pt \
    --max_batch_size 16 \
    --max_new_tokens 256 \
    --temperature 0
