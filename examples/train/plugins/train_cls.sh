# uncomment custom_trainer
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type full \
    --freeze_parameters_ratio 1 \
    --trainable_parameters score \
    --dataset simpleai/HC3-Chinese:baike_cls#1000 \
    --num_train_epochs 1 \
    --num_labels 2 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5
