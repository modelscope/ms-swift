# Experimental environment: 3090
# 10GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python animatediff_sft.py \
    --model_id_or_path output/-2023-11-22T19-59-20/checkpoints \
    --validation_file scripts/validation.txt \
