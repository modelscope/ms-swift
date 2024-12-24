# 4*80GB
# You can refer to `https://github.com/QwenLM/Qwen2-VL` for the meaning of the `VIDEO_MAX_PIXELS` parameter.
nproc_per_node=4

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift sft \
    --model Qwen/QVQ-72B-Preview \
    --dataset swift/VideoChatGPT:all \
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
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3
