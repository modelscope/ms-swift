# 4 * 30GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen3.5-35B-A3B \
    --tuner_type lora \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --experts_impl grouped_mm \
    --router_aux_loss_coef 1e-3 \
    --gradient_accumulation_steps 1 \
    --group_by_length true \
    --output_dir output/Qwen3.5-35B-A3B \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3


# PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# MAX_PIXELS=1003520 \
# VIDEO_MAX_PIXELS=50176 \
# FPS_MAX_FRAMES=12 \
# swift infer \
#     --adapters output/vx-xxx/checkpoint-xxx \
#     --stream true \
#     --experts_impl grouped_mm \
#     --enable_thinking false \
#     --load_data_args true
