# 2 * 48GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron rlhf \
    --rlhf_type kto \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --save_safetensors true \
    --merge_lora false \
    --dataset 'AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto#20000' \
    --load_from_cache_file true \
    --packing true \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --split_dataset_ratio 0.01 \
    --expert_model_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --output_dir megatron_output/Qwen3-30B-A3B-Instruct-2507 \
    --eval_interval 100 \
    --save_interval 100 \
    --max_length 8192 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --beta 0.1 \
    --desirable_weight 1 \
    --undesirable_weight 1
