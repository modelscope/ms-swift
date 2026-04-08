# ms-swift>=3.12
swift export \
    --model Qwen/Qwen3-Reranker-4B \
    --task_type generative_reranker \
    --dataset MTEB/scidocs-reranking \
    --dataset_num_proc 64 \
    --split_dataset_ratio 0.01 \
    --to_cached_dataset true \
    --output_dir ./qwen3_reranker_cached_dataset

# 4 * 24GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen3-Reranker-4B \
    --task_type generative_reranker \
    --loss_type pointwise_reranker \
    --tuner_type full \
    --cached_dataset './qwen3_reranker_cached_dataset/train' \
    --cached_val_dataset './qwen3_reranker_cached_dataset/val' \
    --num_train_epochs 1 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 6e-6 \
    --gradient_accumulation_steps 8 \
    --packing true \
    --eval_steps 50 \
    --save_steps 200 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output/Qwen3-Reranker-4B \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --dataloader_drop_last true
