# test_env: H20, cuda12.9
# FP8: 8 * 58GiB 8s/it
# BF16: 8 * 52GiB 13s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --model Qwen/Qwen3-14B-FP8 \
    --load_safetensors true \
    --save_safetensors true \
    --fp8_recipe blockwise \
    --fp8_format e4m3 \
    --fp8_param_gather true \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT#20000' \
    --load_from_cache_file true \
    --tensor_model_parallel_size 4 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --packing true \
    --recompute_granularity selective \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --cross_entropy_fusion_impl te \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-14B-FP8 \
    --eval_interval 200 \
    --save_interval 200 \
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
