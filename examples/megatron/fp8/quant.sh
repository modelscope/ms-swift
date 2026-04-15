CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3.5-35B-A3B \
    --output_dir Qwen3.5-35B-A3B-FP8 \
    --to_hf true \
    --fp8_recipe blockwise \
    --fp8_format e4m3 \
    --fp8_param_gather true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2
