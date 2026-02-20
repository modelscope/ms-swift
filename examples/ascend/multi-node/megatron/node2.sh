# Atlas A2 * 2 nodes * 8 cards per node

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=xxx.xxx.xxx.xxx \
MASTER_PORT=29500 \
NPROC_PER_NODE=8 \
HCCL_SOCKET_IFNAME=xxx \
megatron sft \
    --model 'Qwen/Qwen3-8B' \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#1000' \
    --output_dir './SAVE' \
    --tuner_type 'lora' \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules 'all-linear' \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 64 \
    --recompute_granularity selective \
    --recompute_modules core_attn \
    --cross_entropy_loss_fusion true \
    --gradient_accumulation_fusion false \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --num_train_epochs 1 \
    --log_interval 5 \
    --dataloader_num_workers 4
