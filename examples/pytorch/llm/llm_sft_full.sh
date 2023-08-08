CUDA_VISIBLE_DEVICES=0 \
python llm_sft.py \
    --model_type qwen-7b \
    --sft_type full \
    --output_dir runs \
    --dataset alpaca-en,alpaca-zh \
    --dataset_sample 20000 \
    --max_length 1024 \
    --quantization_bit 4 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 10 \
