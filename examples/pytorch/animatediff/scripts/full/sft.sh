# Experimental environment: A100 * 4
# 200GB GPU memory totally
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 animatediff_sft.py \
  --model_id_or_path wyj123456/Realistic_Vision_V5.1_noVAE \
  --csv_path /mnt/workspace/yzhao/tastelikefeet/webvid/results_2M_train.csv \
  --video_folder /mnt/workspace/yzhao/tastelikefeet/webvid/videos2 \
  --sft_type full \
  --lr_scheduler_type constant \
  --trainable_modules .*motion_modules.* \
  --batch_size 4 \
  --eval_steps 100 \
  --gradient_accumulation_steps 16 \
