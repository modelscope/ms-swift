# 4*50GiB
# You can refer to `https://github.com/QwenLM/Qwen2-VL` for the meaning of the `MAX_PIXELS` parameter.
# --rlhf_type cpo/orpo/simpo are also supported
nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
MAX_PIXELS=1003520 \
swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dataset 'swift/RLAIF-V-Dataset#20000' \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --deepspeed zero2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
