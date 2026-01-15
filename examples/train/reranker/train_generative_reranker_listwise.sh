# 4*47G
# losses: plugin/loss.py
# only support --padding_side left
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model Qwen/Qwen3-Reranker-4B \
    --task_type generative_reranker \
    --loss_type listwise_reranker \
    --tuner_type full \
    --dataset MTEB/scidocs-reranking \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --padding_side left \
    --eval_strategy steps \
    --output_dir output \
    --eval_steps 100 \
    --num_train_epochs 1 \
    --save_steps 200 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --dataset_num_proc 8 \
    --learning_rate 6e-6 \
    --label_names labels \
    --deepspeed zero2 \
    --dataloader_drop_last true
