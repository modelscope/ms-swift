# 8 * 80GiB
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --load Qwen3-32B-mcore \
    --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT' \
    --tensor_model_parallel_size 8 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --train_iters 10000 \
    --max_epochs 5 \
    --eval_iters 50 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 100 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-32B \
    --eval_interval 500 \
    --save_interval 500 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash
