# 2 * 70GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron rlhf \
    --rlhf_type rm \
    --load Qwen2.5-VL-7B-Instruct-mcore \
    --dataset 'swift/RLAIF-V-Dataset#20000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 2 \
    --packing true \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen2.5-VL-7B-Instruct \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --center_rewards_coefficient 0.01
