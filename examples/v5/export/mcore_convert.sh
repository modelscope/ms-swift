# v5 export (megatron backend): convert weights between HF safetensors and Megatron-core (mcore) torch_dist.
#
# This is `swift export --backend megatron`: v5 has no separate `megatron export` command, the backend is
# a single flag that pins DistributedConfig.backend='megatron' and reuses the same export pipeline.
# Two directions, run as separate commands:
#   --to_mcore : HF safetensors -> mcore torch_dist (the layout megatron training reads)
#   --to_hf    : mcore torch_dist -> HF safetensors (back to a transformers-loadable model)
# The parallelism sizes (--tensor_model_parallel_size / --pipeline_model_parallel_size /
# --expert_model_parallel_size) come from DistributedConfig and DEFINE the mcore checkpoint layout, so a
# to_hf must use a layout that can read the torch_dist shards. --mcore_model is the source for to_hf.
#
# --test_convert_precision loads both models and compares their outputs on the same input (costs a second
# model in memory, hence opt-in). to_hf always writes safetensors (--safe_serialization defaults true).
# Launch under torchrun: NPROC_PER_NODE drives the world size.

# safetensors -> torch_dist
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
USE_SWIFT_V5=1 \
swift export \
    --backend megatron \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --to_mcore true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true \
    --output_dir Qwen3-30B-A3B-Instruct-2507-mcore

# torch_dist -> safetensors
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
USE_SWIFT_V5=1 \
swift export \
    --backend megatron \
    --mcore_model Qwen3-30B-A3B-Instruct-2507-mcore \
    --to_hf true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true \
    --output_dir Qwen3-30B-A3B-Instruct-2507-hf
