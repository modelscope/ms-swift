# 4 * 28GiB; 3.4s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron rlhf \
    --rlhf_type dpo \
    --load Qwen3-4B-Instruct-2507-mcore \
    --dataset 'AI-ModelScope/orpo-dpo-mix-40k' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 4 \
    --packing true \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-4B-Instruct-2507 \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --beta 0.1 \
    --loss_type sigmoid
