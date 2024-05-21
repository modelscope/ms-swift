# Experimental environment: 2 * A100
# 2 * 55GB GPU memory
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model_type qwen-vl-chat \
    --sft_type full \
    --train_dataset_sample -1 \
    --eval_steps 100 \
    --output_dir output \
    --num_train_epochs 1 \
    --max_length 2048 \
    --learning_rate 1e-5 \
    --use_flash_attn true \
    --save_only_model true \
    --dataset coco-en-mini \
