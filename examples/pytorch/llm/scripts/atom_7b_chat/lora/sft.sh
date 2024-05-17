# Experimental environment: 3090，A10，V100...
# 20GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type atom-7b-chat \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --dtype AUTO \
    --output_dir output \
    --ddp_backend nccl \
    --dataset ms-bench \
    --num_train_epochs 3 \
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
