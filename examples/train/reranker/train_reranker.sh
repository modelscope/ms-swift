# 1*5G
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model iic/gte-reranker-modernbert-base \
    --task_type reranker \
    --loss_type reranker \
    --train_type full \
    --dataset MTEB/scidocs-reranking \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir output \
    --eval_steps 100 \
    --num_train_epochs 1 \
    --save_steps 200 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --dataset_num_proc 8 \
    --learning_rate 6e-6 \
    --label_names labels \
    --dataloader_drop_last true \
