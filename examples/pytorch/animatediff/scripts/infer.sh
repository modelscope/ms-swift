# Experimental environment: 3090
# 10GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python animatediff_infer.py \
  --model_id_or_path wyj123456/Realistic_Vision_V5.1_noVAE \
  --sft_type full \
  --motion_adapter_id_or_path Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2 \
  --ckpt_dir /mnt/workspace/yzhao/tastelikefeet/swift/examples/pytorch/animatediff/output/ad-2023-11-25T15-13-58/checkpoints/iter-150000 \
  --eval_human true  \
