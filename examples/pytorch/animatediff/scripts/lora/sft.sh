# Experimental environment: A100
# 20GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python animatediff_sft.py \
  --model_id_or_path wyj123456/Realistic_Vision_V5.1_noVAE \
  --csv_path /mnt/workspace/yzhao/tastelikefeet/webvid/results_2M_train.csv \
  --video_folder /mnt/workspace/yzhao/tastelikefeet/webvid/videos2 \
  --motion_adapter_id_or_path Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2 \
  --sft_type lora \
  --lr_scheduler_type constant \
  --trainable_modules .*motion_modules.* \
  --batch_size 1 \
  --eval_steps 200 \
  --dataset_sample_size 10000 \
  --gradient_accumulation_steps 16 \
