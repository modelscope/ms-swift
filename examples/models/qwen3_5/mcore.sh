# Packing is not recommended due to the presence of linear_attention
# 4 * 40GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
megatron sft \
    --model Qwen/Qwen3.5-35B-A3B \
    --save_safetensors true \
    --merge_lora true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --expert_model_parallel_size 4 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 4 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --group_by_length true \
    --finetune true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --output_dir megatron_output/Qwen3.5-35B-A3B \
    --eval_steps 200 \
    --save_steps 200 \
    --max_length 2048 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend unfused \
    --padding_free false \
    --model_author swift \
    --model_name swift-robot


# PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# MAX_PIXELS=1003520 \
# VIDEO_MAX_PIXELS=50176 \
# FPS_MAX_FRAMES=12 \
# swift infer \
#     --model megatron_output/vx-xxx/checkpoint-xxx-merged \
#     --stream true \
#     --experts_impl grouped_mm \
#     --enable_thinking false \
#     --load_data_args true
