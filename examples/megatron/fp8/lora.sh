# 2 * 60GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-32B-FP8 \
    --load_safetensors true \
    --save_safetensors true \
    --tuner_type lora \
    --lora_rank 128 \
    --lora_alpha 256 \
    --target_modules all-linear \
    --merge_lora true \
    --fp8_recipe blockwise \
    --fp8_format e4m3 \
    --fp8_param_gather true \
    --dataset 'swift/Qwen3-SFT-Mixin#2000' \
              'swift/self-cognition:empty_think#600' \
    --load_from_cache_file true \
    --tensor_model_parallel_size 2 \
    --micro_batch_size 8 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --cross_entropy_fusion_impl te \
    --loss_scale ignore_empty_think \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-32B-FP8 \
    --eval_interval 100 \
    --save_interval 100 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --use_precision_aware_optimizer true \
    --exp_avg_dtype bf16 \
    --exp_avg_sq_dtype bf16 \
    --attention_backend flash
