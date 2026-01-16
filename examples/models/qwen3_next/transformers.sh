# 4 * 60GiB
# Note: Due to linear attention, this model currently does not support padding_free and packing.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup swift sft \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --train_type lora \
    --dataset 'swift/self-cognition#1000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --router_aux_loss_coef 1e-3 \
    --gradient_accumulation_steps 2 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
