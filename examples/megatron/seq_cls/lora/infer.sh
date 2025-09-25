# acc: 0.9298597194388778
# 60GiB
CUDA_VISIBLE_DEVICES=0 \
swift infer \
  --model megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx-hf \
  --load_data_args true \
  --max_batch_size 16 \
  --attn_impl flash_attn \
  --metric acc
