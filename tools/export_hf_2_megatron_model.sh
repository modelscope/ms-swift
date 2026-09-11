export MEGATRON_LM_PATH='/data_large_v2/liangxiaoyun/projects/Megatron-LM'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift export \
    --model /data_large_v2/liangxiaoyun/model_output/Qwen2.5-72B-Instruct \
    --output_dir /data_large_v2/liangxiaoyun/model_output/Qwen2.5-72B-Instruct-megatron-v2 \
    --to_mcore true \
    --torch_dtype bfloat16


# export MEGATRON_LM_PATH='/data_large_v2/liangxiaoyun/projects/Megatron-LM'

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# NPROC_PER_NODE=8 \
# swift export \
#     --model /data_large_v2/liangxiaoyun/model_output/Qwen3-32B \
#     --output_dir /data_large_v2/liangxiaoyun/model_output/Qwen3-32B-megatron \
#     --to_mcore true \
#     --torch_dtype bfloat16
