# 4 * 50GiB 14s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron rlhf \
    --rlhf_type dpo \
    --load Qwen2.5-VL-7B-Instruct-mcore \
    --dataset 'swift/RLAIF-V-Dataset#20000' \
    --load_from_cache_file true \
    --train_type full \
    --tensor_model_parallel_size 4 \
    --sequence_parallel true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --packing true \
    --split_dataset_ratio 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --max_epochs 1 \
    --save megatron_output/Qwen2.5-VL-7B-Instruct \
    --save_interval 200 \
    --vit_gradient_checkpointing true \
    --max_length 8192 \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 8 \
    --attention_backend flash \
    --beta 0.1 \
    --loss_type sigmoid
