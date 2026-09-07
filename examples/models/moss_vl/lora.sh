# 1 * 80GiB
# MOSS-VL requires non-reentrant gradient checkpointing and does not support packing/padding-free training.
pip install "transformers>=4.57.1,<5" joblib
pip install torchcodec==0.7.0  # match your torch version: 0.7.x for torch 2.8

VIDEO_MIN_PIXELS=256 \
VIDEO_MAX_PIXELS=16384 \
FPS=1 \
FPS_MAX_FRAMES=256 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model OpenMOSS-Team/MOSS-VL-Instruct-0708 \
    --dataset 'lmms-lab/VideoChatGPT:Generic#1000' \
    --use_hf true \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl eager \
    --packing false \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --max_pixels 262144 \
    --output_dir output/moss_vl_lora \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4
