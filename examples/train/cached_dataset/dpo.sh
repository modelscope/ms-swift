# ms-swift>=3.11
OMP_NUM_THREADS=14 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
swift export \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --dataset swift/RLAIF-V-Dataset \
    --split_dataset_ratio 0.01 \
    --dataset_num_proc 8 \
    --to_cached_dataset true \
    --template_mode rlhf \
    --output_dir ./qwen3_vl_cached_dataset


# 16s/it; 8 * 65GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
megatron rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --cached_dataset qwen3_vl_cached_dataset/train \
    --cached_val_dataset qwen3_vl_cached_dataset/val \
    --load_from_cache_file true \
    --train_type full \
    --tensor_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-VL-30B-A3B-Instruct \
    --eval_interval 500 \
    --save_interval 500 \
    --max_length 8192 \
    --max_epochs 1 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer true \
    --optimizer_offload_fraction 0.4 \
    --attention_backend flash \
    --rpo_alpha 0.1 \
    --beta 0.1 \
    --loss_type sigmoid
