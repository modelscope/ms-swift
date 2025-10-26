# 显存配置与并行设置
# PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
# IMAGE_MAX_TOKEN_NUM=1024
# NPROC_PER_NODE=2 # GPU available for training
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 # GPU ID for training \
# WANDB_PROJECT='Qwen3-VL-Robotics' # WandB project name
# WANDB_RUN_NAME='Qwen3-VL-4B-Instruct-$(date +%Y%m%d_%H%M%S)' # WandB run name
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export IMAGE_MAX_TOKEN_NUM=1024
export NPROC_PER_NODE=2
export ROOT_IMAGE_DIR=/home/yuquan002/ssd/libero_vl_dataset
export WANDB_PROJECT='Qwen3-VL-Robotics'
export WANDB_RUN_NAME='Qwen3-VL-4B-Instruct-Spatial-$(date +%Y%m%d)'
swift sft \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --custom_register_path /home/yuquan002/ssd/ms-swift-robotics/vla-0/data/libero_dataset.py \
    --dataset libero-spatial-vla0 \
    --split_dataset_ratio 0.01 \
    --load_from_cache_file true \
    --report_to tensorboard wandb \
    --use_hf true \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 12 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --attn_impl flash_attn \
    --padding_free true \
    --learning_rate 5e-6 \
    --packing false \
    --gradient_checkpointing true \
    --gradient_accumulation_steps 16 \
    --eval_steps 500 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --max_length 2048 \
    --output_dir output/qwen3-vl-4b-instruct-vla0-libero \
    --warmup_ratio 0.03 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 8 \
    # --deepspeed zero2 \

# TODO: Resume Training from Checkpoint, and load trained weights 