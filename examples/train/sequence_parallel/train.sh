# 27.5GiB * 2
nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/LongAlpaca-12k#5000' \
    --num_train_epochs 1 \
    --sequence_parallel_size 2 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --lora_rank 8 \
    --lora_alpha 32 \
    --eval_steps 100 \
    --save_steps 100 \
    --max_length 10000 \
    --save_total_limit 2 \
    --logging_steps 5
