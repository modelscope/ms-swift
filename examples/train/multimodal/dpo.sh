# --rlhf_type cpo/orpo/simpo are also supported
nproc_per_node=4

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --train_type lora \
    --dataset swift/RLAIF-V-Dataset \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --deepspeed zero3 \
    --logging_steps 5
