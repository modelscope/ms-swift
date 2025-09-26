PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron rlhf \
    --rlhf_type kto \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --load Qwen2.5-7B-Instruct-mcore \
    --dataset 'AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto#10000' \
    --split_dataset_ratio 0 \
    --micro_batch_size 8 \
    --global_batch_size 16 \
    --max_epochs 1 \
    --finetune true \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save megatron_output/Qwen2.5-7B-Instruct \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 2048 \
    --num_workers 8 \
    --dataset_num_proc 16 \
    --beta 0.1 \
    --desirable_weight 1 \
    --undesirable_weight 1 \
    --calculate_KL true \
    --log_interval 5
