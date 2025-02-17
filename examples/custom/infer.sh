# sh examples/custom/infer.sh
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --load_data_args true \
    --infer_backend pt \
    --max_batch_size 16 \
    --max_new_tokens 256 \
    --temperature 0
