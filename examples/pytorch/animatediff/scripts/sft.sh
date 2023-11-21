# Experimental environment: 3090
# 10GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python animatediff_sft.py \
    --csv_path /mnt/workspace/yzhao/tastelikefeet/Webcam.csv \
    --video_folder /mnt/workspace/yzhao/tastelikefeet/videos \
    --sft_type full \
