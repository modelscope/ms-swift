export USE_SWIFT_V5=0
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model LiquidAI/LFM3-1.7B \
    --model_type lfm3 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
    --tuner_type lora \
    --target_modules all-linear \
    --max_length 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --attn_impl flash_attention_2 \
    --output_dir output/lfm3_sft
