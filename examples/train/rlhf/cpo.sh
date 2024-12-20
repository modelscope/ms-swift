nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type cpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji:zh \
    --num_train_epochs 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5
