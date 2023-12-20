# Experimental environment: A100
# 18GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python animatediff_infer.py \
  --model_id_or_path wyj123456/Realistic_Vision_V5.1_noVAE \
  --sft_type full \
  --ckpt_dir /output/path/like/checkpoints/iter-xxx \
