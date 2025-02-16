#!/bin/bash
if [ "$RANK" -eq "0" ]; then
    NPROC=1
else
    NPROC=2
fi

PYTHONPATH=. \
torchrun \
    --nproc_per_node=${NPROC} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    swift/cli/rlft.py \
    --rlhf_type grpo \
    --model /mnt/nas3/.cache/modelscope/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --train_type lora \
    --dataset AI-MO/NuminaMath-TIR#10000 \
    --torch_dtype bfloat16 \
    --system examples/train/grpo/prompt.txt \
    --num_train_epochs 1 \
    --max_length 2048 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 2e-6 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_completion_length 2048 \
    --reward_funcs format \
    --num_generations 2 \
    --use_vllm true \
    --deepspeed zero3 \
    --vllm_gpu_memory_utilization 0.7