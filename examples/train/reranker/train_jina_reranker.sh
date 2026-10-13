# jina-reranker-v3.5 uses the same integration; replace the model id below to use it.
USE_HF=1 CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model jinaai/jina-reranker-v3 \
    --task_type reranker \
    --loss_type listwise_reranker \
    --tuner_type lora \
    --dataset MTEB/scidocs-reranking \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir output \
    --eval_steps 100 \
    --num_train_epochs 1 \
    --save_steps 200 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --dataset_num_proc 8 \
    --learning_rate 1e-4 \
    --max_length 2048 \
    --label_names labels \
    --dataloader_drop_last true
