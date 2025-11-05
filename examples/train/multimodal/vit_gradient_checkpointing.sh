# gc true, vgc true: 48GiB, 2.45s/it
# gc true, vgc false: 62GiB 2.32s/it
# gc false, vgc true: 56GiB 2.16s/it
# gc false, vgc false: 77GiB 1.95s/it
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset swift/VideoChatGPT:all \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attn \
    --padding_free true \
    --save_only_model true
