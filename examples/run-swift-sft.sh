# Experimental environment: A100
# 80GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type internvl2-1b \
    --model_id_or_path /mnt/n/model/OpenGVLab/InternVL2-1B \
    --dataset coco-en-2-mini \
    --max_length 4096