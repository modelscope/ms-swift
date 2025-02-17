# 17.3GiB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type bone \
    --label_names labels \
    --dataset 'swift/self-cognition#1000' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --model_author swift \
    --model_name swift-robot
