nproc_per_node=8
# 4*12G
# losses: plugin/loss.py
# data format: docs/source_en/Customization/Custom-dataset.md
# --use_chat_template must be false to use generation template
# --dataloader_drop_last must be true or eval gather will throw error
# --model iic/gte-modernbert-base modernbert also supported
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model iic/gte_Qwen2-7B-instruct \
    --train_type lora \
    --dataset 'sentence-transformers/stsb' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --eval_strategy steps \
    --use_chat_template false \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --learning_rate 5e-6 \
    --deepspeed zero3 \
    --dataloader_num_workers 4 \
    --task_type embedding \
    --loss_type cosine_similarity \
    --dataloader_drop_last true \
