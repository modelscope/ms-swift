# Experimental environment: A10
# 12GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type phi3-4b-4k-instruct \
    --sft_type lora \
    --template_type AUTO \
    --train_dataset_sample 20000 \
    --eval_steps 100 \
    --output_dir output \
    --num_train_epochs 3 \
    --max_length 4096 \
    --learning_rate 1e-4 \
    --use_flash_attn false \
    --lora_target_modules ALL \
    --dataset codefuse-python-en \
