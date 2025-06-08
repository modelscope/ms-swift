nproc_per_node=8
# 4*12G
# losses: plugin/loss.py
# data format: docs/source_en/Customization/Custom-dataset.md
# --use_chat_template must be false to use generation template
# --dataloader_drop_last must be true or eval gather will throw error
# --model iic/gte-modernbert-base iic/gte_Qwen2-7B-instruct also supported
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model Qwen/Qwen3-Embedding-0.6B \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type full \
    --dataset sentence-transformers/stsb:positive \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir output \
    --eval_steps 20 \
    --num_train_epochs 5 \
    --save_steps 70 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero3
