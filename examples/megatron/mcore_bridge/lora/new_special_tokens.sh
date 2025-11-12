# 2 * 50GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --load_safetensors true \
    --save_safetensors true \
    --merge_lora false \
    --dataset 'swift/new_special_tokens' \
    --load_from_cache_file true \
    --new_special_tokens 'examples/train/new_special_tokens/tokens.txt' \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --modules_to_save word_embeddings output_layer \
    --split_dataset_ratio 0.01 \
    --expert_model_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 32 \
    --global_batch_size 64 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 5 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507 \
    --eval_interval 500 \
    --save_interval 500 \
    --max_length 2048 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --moe_expert_capacity_factor 2 \
    --sequence_parallel true \
    --attention_backend flash

# CUDA_VISIBLE_DEVICES=0,1 \
# swift infer \
#     --adapters megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx \
#     --load_data_args true \
#     --stream true
