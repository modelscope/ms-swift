CUDA_VISIBLE_DEVICES=0,1 \
python llm_sft.py \
    --model_type qwen-7b \
    --sft_type lora \
    --output_dir runs \
    --dataset alpaca-en,alpaca-zh \
    --dataset_sample 20000 \
    --max_length 1024 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.1 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
