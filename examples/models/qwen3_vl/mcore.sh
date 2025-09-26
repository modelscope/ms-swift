# 8 * 80GiB; 45min
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
IMAGE_MAX_TOKEN_NUM=1024 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --load Qwen3-VL-235B-A22B-Instruct-mcore \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#20000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --moe_permute_fusion true \
    --tensor_model_parallel_size 2 \
    --expert_tensor_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save megatron_output/Qwen3-VL-235B-A22B-Instruct \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 2048 \
    --packing true \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash
