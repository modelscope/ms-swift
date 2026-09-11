export MEGATRON_LM_PATH='/data_large_v2/liangxiaoyun/projects/Megatron-LM'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift export \
    --mcore_model /data_large_v2/liangxiaoyun/model_output/agent/Qwen3-32B-Instruct-agent-64K-sft_megatron_1208/v32-20251211-110721 \
    --output_dir /data_large_v2/liangxiaoyun/model_output/agent/Qwen3-32B-Instruct-agent-64K-sft_megatron_1208/v32-20251211-110721/iter_0000800_hf \
    --to_hf true \
    --torch_dtype bfloat16 \
    --model_type qwen3 \
    --model /data_large_v2/liangxiaoyun/model_output/Qwen3-32B
