# 44G
# llama2 is not good at Chinese
CUDA_VISIBLE_DEVICES=0,1 \
python src/llm_sft.py \
    --model_type llama2-70b-chat \
    --sft_type lora \
    --output_dir runs \
    --dataset alpaca-en \
    --dataset_sample 20000 \
    --num_train_epochs 1 \
    --max_length 1024 \
    --quantization_bit 4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.1 \
    --gradient_checkpointing true \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 10 \
