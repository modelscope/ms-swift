# Experimental environment: A10
# 7GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type yuan2-2b-janus-instruct \
    --sft_type lora \
    --template_type AUTO \
    --dataset hc3-zh \
    --train_dataset_sample 20000 \
    --eval_steps 100 \
    --output_dir output \
    --num_train_epochs 1 \
    --max_length 2048 \
    --learning_rate 1e-4 \
    --use_flash_attn false \
    --lora_target_modules ALL \
