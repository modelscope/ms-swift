# Train
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2-1.5B-Instruct \
    --train_type lora \
    --dataset swift/self-cognition#1000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --init_weights lora-ga \
    --lora_ga_batch_size 2 \
    --lora_ga_iters 2 \
    --lora_ga_max_length 1024 \
    --lora_ga_direction ArB2r \
    --lora_ga_scale stable \
    --lora_ga_stable_gamma 16 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --model_author swift \
    --model_name swift-robot

# Infer
# swift infer \
#     --model Qwen/Qwen2-1.5B-Instruct \
#     --ckpt_dir ./output/Qwen2-1.5B-Instruct/v0-20241214-191235/checkpoint-62/converted/default \
#     --infer_backend pt \
#     --stream true
