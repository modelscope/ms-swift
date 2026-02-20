# 4 * 80GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron pt \
    --model Qwen/Qwen2.5-7B \
    --save_safetensors true \
    --dataset swift/chinese-c4 \
    --streaming true \
    --packing true \
    --tensor_model_parallel_size 4 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity selective \
    --train_iters 10000 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 300 \
    --min_lr 1e-6 \
    --output_dir megatron_output/Qwen2.5-7B \
    --eval_interval 500 \
    --save_interval 500 \
    --max_length 8192 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash
