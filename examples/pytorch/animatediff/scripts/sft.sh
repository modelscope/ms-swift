# Experimental environment: 3090
# 10GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 animatediff_sft.py \
    --base_model_id_or_path AI-ModelScope/stable-diffusion-v1-5 \
    --csv_path /mnt/workspace/yzhao/tastelikefeet/webvid/results_2M_train.csv \
    --video_folder /mnt/workspace/yzhao/tastelikefeet/webvid/videos2 \
    --sft_type full \
    --lr_scheduler_type constant \
    --trainable_modules .*motion_modules.* \
    --batch_size 4 \
    --eval_steps 1000 \
    --gradient_accumulation_steps 1 \
    --use_wandb true \
