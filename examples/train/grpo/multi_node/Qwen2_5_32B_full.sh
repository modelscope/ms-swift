# External vLLM

# Assume we have two nodes, one with 8 GPUs of 80GB each (880G) and another with 2 GPUs of 80GB each (2 80G).
#   NODE1. The node with 2*80G will be used to deploy the vLLM server.
#   NODE2. The node with 8*80G will be used for full-parameter fine-tuning of the 32B model.

# Note : Use beta=0 to disable the reference model; otherwise, it may lead to Out-of-Memory (OOM) errors.

# NODE1 for vLLM Server
CUDA_VISIBLE_DEVICES=0,1 \
swift deploy \
    --model Qwen/Qwen2.5-32B-Instruct \
    --infer_backend vllm \
    --use_async_engine false \
    --tensor_parallel_size 2

# NODE2 for Training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-32B-Instruct \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_server_host xxx \
    --vllm_server_port 8000 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset AI-MO/NuminaMath-TIR#1000 \
    --max_completion_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --deepspeed zero3 \
    --log_completions true \
    --num_iterations 1 \
    --num_infer_workers 1 \
    --report_to tensorboard wandb \
    --beta 0.0
