# 27.5GiB * 2
nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset swift/self-cognition#1000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --model_author swift \
    --model_name swift-robot
