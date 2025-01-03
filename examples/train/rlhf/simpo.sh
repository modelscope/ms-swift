nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type simpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type full \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji:zh \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --deepspeed zero3 \
    --logging_steps 5
