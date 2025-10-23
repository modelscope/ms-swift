nproc_per_node=2
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1 \
MAX_NEGATIVE_SAMPLES=1 \
MAX_PIXELS=602112 \
swift sft \
    --model JinaAI/jina-reranker-m0 \
    --task_type reranker \
    --loss_type reranker \
    --train_type lora \
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
    --learning_rate 6e-6 \
    --label_names labels \
    --dataloader_drop_last true \
    --attn_impl flash_attn \
    --padding_free true
