nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type dpo \
    --model_id_or_path LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --sft_type lora \
    --dataset ultrafeedback-kto \
    --streaming true \
    --num_train_epochs 1 \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5