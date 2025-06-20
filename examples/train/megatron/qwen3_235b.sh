# ppue 32 * 16 = 512 gpus
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
NNODES=$NNODES \
NODE_RANK=$NODE_RANK \
NPROC_PER_NODE=$NPROC_PER_NODE \        
megatron sft \
    --load /mnt/sllmworkspace-p/common/sllmworks/model/Qwen3-235B-A22B-mcore \
    --dataset /sllmworks/datasets/Chinese-DeepSeek-R1-Distill-data-110k-SFT \
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 8 \
    --pipeline_model_parallel_size 2 \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 0.01 \
    --micro_batch_size 2 \
    --global_batch_size 128 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --train_iters 20 \
    --split_dataset_ratio 0 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-05 \
    --lr_warmup_iters 2 \
    --min_lr 1e-06 \
    --save /mnt/sllmworkspace-p/common/sllmworks/2641825084/checkpoint \
    --save_interval 200 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --use_flash_attn false \
    --attention_backend auto \
    --tensorboard_dir /home/admin/logs/tfevent \
    --max_epochs 1 \
    --ckpt_format torch_dist