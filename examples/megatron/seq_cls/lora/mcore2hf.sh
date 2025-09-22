CUDA_VISIBLE_DEVICES=0,1 \
swift export \
    --mcore_adapters megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx-hf \
    --test_convert_precision true
