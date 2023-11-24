# Experimental environment: 3090
# 10GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 animatediff_sft.py \
  --model_id_or_path wyj123456/Realistic_Vision_V5.1_noVAE \
  --csv_path /mnt/workspace/yzhao/tastelikefeet/webvid/results_2M_train.csv \
  --video_folder /mnt/workspace/yzhao/tastelikefeet/webvid/videos2 \
  --sft_type lora \
  --lr_scheduler_type constant \
  --trainable_modules .*motion_modules.* \
  --batch_size 1 \
  --eval_steps 30 \
  --gradient_accumulation_steps 16 \
  --motion_adapter_id_or_path Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2 \
