NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
MAX_NEGATIVE_SAMPLES=1 \
swift sft \
    --model JinaAI/jina-reranker-m0 \
    --task_type reranker \
    --loss_type pointwise_reranker \
    --tuner_type lora \
    --dataset swift/TextCaps:rerank \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir output \
    --eval_steps 100 \
    --num_train_epochs 1 \
    --save_steps 200 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --dataset_num_proc 8 \
    --learning_rate 6e-5 \
    --label_names labels \
    --dataloader_drop_last true \
    --attn_impl flash_attn \
    --padding_free true
