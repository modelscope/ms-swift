# Experimental environment: 4 * A100
# 4 * 60GB GPU memory
# Note: you have to install latest version of the transformers library.
# pip install git+https://github.com/huggingface/transformers
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type c4ai-command-r-plus \
    --sft_type lora \
    --tuner_backend swift \
    --dtype AUTO \
    --output_dir output \
    --dataset alpaca-zh#10000 alpaca-en#10000 \
    --num_train_epochs 2 \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules DEFAULT \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn true \
