# 22GiB
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=602112 \
swift sft \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --train_type lora \
    --dataset swift/OK-VQA_train#1000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5
