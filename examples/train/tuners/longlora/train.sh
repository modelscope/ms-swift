CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --train_type longlora \
    --dataset 'AI-ModelScope/LongAlpaca-12k#1000' \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --attn_impl flash_attn \
    --gradient_accumulation_steps 16 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --eval_steps 100 \
    --save_steps 100 \
    --max_length 10000 \
    --save_total_limit 2 \
    --logging_steps 5
