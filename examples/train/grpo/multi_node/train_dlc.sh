NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
PYTHONPATH=. \
torchrun \
    --nproc_per_node=8 \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    swift/cli/rlhf.py \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B \
    --train_type full \
    --dataset AI-MO/NuminaMath-TIR#10000 \
    --torch_dtype bfloat16 \
    --system examples/train/grpo/prompt.txt \
    --num_train_epochs 1 \
    --max_length 2048 \
    --vllm_max_model_len 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_completion_length 2048 \
    --reward_funcs accuracy format \
    --num_generations 48 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.3 \
    --sleep_level 1 \
    --deepspeed zero3_offload \
    --num_infer_workers 8 \
    --tensor_parallel_size 4 \
    --temperature 1.0 \
    --top_p 0.85
