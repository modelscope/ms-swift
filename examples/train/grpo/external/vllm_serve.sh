CUDA_VISIBLE_DEVICES=0,1 \
swift rollout \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tensor_parallel_size 2
