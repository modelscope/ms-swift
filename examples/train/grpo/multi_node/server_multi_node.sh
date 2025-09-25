# NOTE: Requires NCCL connectivity between the training master node and rollout nodes
# This script demonstrates multi-node rollout and multi-node training with swift.
# node1 and node2: multi-node rollout servers
# node3 and node4: distributed training nodes

# --- Rollout Section ---
# For rollout, you can launch any number of servers on different nodes
# Start rollout server on node1:
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rollout \
    --model Qwen/Qwen2.5-7B-Instruct \
    --vllm_tensor_parallel_size 2 \
    --vllm_data_parallel_size 2 \
    --port <node1_port>

# Start rollout server on node2:
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rollout \
    --model Qwen/Qwen2.5-7B-Instruct \
    --vllm_tensor_parallel_size 2 \
    --vllm_data_parallel_size 2 \
    --port <node2_port>

# --- Training Section ---
# node3: Master training node (rank 0)
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host <node1_ip> <node2_ip> \
    --vllm_server_port <node1_port> <node2_port> \
    --dataset AI-MO/NuminaMath-TIR#1000 \
    --load_from_cache_file true \
    --max_completion_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --deepspeed zero2 \
    --log_completions true \

# node4: Secondary training node (rank 1)
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=<node3_ip> \
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host <node1_ip> <node2_ip> \
    --vllm_server_port <node1_port> <node2_port> \
    --dataset AI-MO/NuminaMath-TIR#1000 \
    --load_from_cache_file true \
    --max_completion_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --deepspeed zero2 \
    --log_completions true \
