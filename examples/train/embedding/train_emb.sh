nproc_per_node=2
# 2*12G
# losses: plugin/loss.py
# data format: docs/source_en/Customization/Custom-dataset.md
# --use_chat_template must be false to use generation template
# --dataloader_drop_last must be true or eval gather will throw error
# --model iic/gte-modernbert-base iic/gte_Qwen2-7B-instruct also supported
# INFONCE_TEMPERATURE default value is 0.01, here we use 0.1 because it makes
# the `sentence-transformers/stsb:positive` dataset result to a zero loss
CUDA_VISIBLE_DEVICES=0,1 \
INFONCE_TEMPERATURE=0.1 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model Qwen/Qwen3-Embedding-0.6B \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type full \
    --dataset sentence-transformers/stsb:positive \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir output \
    --save_steps 50 \
    --eval_steps 50 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero2
