# Experimental environment: A100
# 60GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type phi2-3b \
    --sft_type lora \
    --template_type default \
    --train_dataset_sample 20000 \
    --eval_steps 100 \
    --output_dir output \
    --num_train_epochs 1 \
    --max_length 2048 \
    --learning_rate 1e-4 \
    --use_flash_attn true \
    --lora_target_modules ALL \
    --dataset codefuse-python-en \
    --gradient_checkpointing false \
