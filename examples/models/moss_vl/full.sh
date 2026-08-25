# 8 * 80GiB
# MOSS-VL requires non-reentrant gradient checkpointing and does not support packing/padding-free training.
pip install "transformers>=4.57.1,<5" joblib
pip install torchcodec==0.7.0  # match your torch version: 0.7.x for torch 2.8

VIDEO_MIN_PIXELS=256 \
VIDEO_MAX_PIXELS=16384 \
FPS=1 \
FPS_MAX_FRAMES=256 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model OpenMOSS-Team/MOSS-VL-Instruct-0708 \
    --dataset 'lmms-lab/VideoChatGPT:Generic#1000' \
    --use_hf true \
    --split_dataset_ratio 0.01 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl eager \
    --packing false \
    --learning_rate 1e-5 \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --max_pixels 262144 \
    --output_dir output/moss_vl_full \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4
