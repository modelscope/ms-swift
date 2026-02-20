# safetensors -> torch_dist
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --output_dir Qwen3-30B-A3B-Instruct-2507-mcore \
    --to_mcore true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true

# torch_dist -> safetensors
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --mcore_model Qwen3-30B-A3B-Instruct-2507-mcore \
    --output_dir Qwen3-30B-A3B-Instruct-2507-hf \
    --to_hf true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true
