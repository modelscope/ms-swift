# 2 * 15GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --merge_lora false \
    --dataset 'tany0699/garbage265#20000' \
    --load_from_cache_file true \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --packing true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --split_dataset_ratio 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_epochs 1 \
    --save megatron_output/Qwen3-VL-8B-Instruct \
    --save_interval 200 \
    --vit_gradient_checkpointing false \
    --max_length 2048 \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --num_labels 265 \
    --task_type seq_cls \
    --dataset_num_proc 8

# PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
# IMAGE_MAX_TOKEN_NUM=1024 \
# VIDEO_MAX_TOKEN_NUM=128 \
# FPS_MAX_FRAMES=16 \
# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --adapters megatron_output/Qwen3-VL-8B-Instruct/vx-xxx \
#     --load_data_args true \
#     --stream true
