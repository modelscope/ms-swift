nproc_per_node=8

NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model iic/gte-modernbert-base \
    --train_type full \
    --dataset 'sentence-transformers/stsb' \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --eval_strategy steps \
    --use_chat_template false \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --learning_rate 5e-6 \
    --deepspeed zero3 \
    --dataloader_num_workers 4 \
    --task_type embedding \
    --loss_type cosent \
    --label_names labels \
    --metric_for_best_model cosine \
    --greater_is_better true \
    --dataloader_drop_last true \
