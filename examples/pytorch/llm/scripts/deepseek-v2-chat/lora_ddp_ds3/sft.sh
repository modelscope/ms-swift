# Experimental environment: 8*A100
# 8*80GB GPU memory
NPROC_PER_NODE=8 \
swift sft \
    --model_type deepseek-v2-chat \
    --sft_type lora \
    --tuner_backend peft \
    --dtype bf16 \
    --output_dir output \
    --ddp_backend nccl \
    --dataset self-cognition#1000 \
    --model_name 小白 'Xiao Bai' \
    --model_author 魔搭 'Modelscope' \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 512 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_dtype AUTO \
    --lora_target_modules DEFAULT \
    --gradient_checkpointing false \
    --batch_size 2 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2\
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --deepspeed default-zero3 \
