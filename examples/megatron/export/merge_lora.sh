NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron export \
    --adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx \
    --tensor_model_parallel_size 2 \
    --to_hf true \
    --merge_lora true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-merged
