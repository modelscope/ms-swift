# Note: cached_dataset does not support CP temporarily.
swift export \
    --model Qwen/Qwen3-30B-A3B-Base \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT' \
    --max_length 8192 \
    --split_dataset_ratio 0.01 \
    --dataset_num_proc 64 \
    --to_cached_dataset true \
    --output_dir ./qwen3_cached_dataset


# 4 * 48GiB; 17s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron sft \
    --load Qwen3-30B-A3B-Base-mcore \
    --cached_dataset './qwen3_cached_dataset' \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --split_dataset_ratio 0.01 \
    --moe_permute_fusion true \
    --expert_model_parallel_size 4 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 3 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save megatron_output/Qwen3-30B-A3B-Base \
    --eval_interval 200 \
    --save_interval 200 \
    --packing true \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash
