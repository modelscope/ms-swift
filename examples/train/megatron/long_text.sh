# Env: 4 * A100
# https://github.com/modelscope/ms-swift/blob/main/examples/train/long_text/zero3.sh
# Max Length: 32K
# GPU Memory: 4 * 50GB, Training Speed 23s/it
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron sft \
    --load Qwen2.5-7B-mcore \
    --dataset 'ZhipuAI/LongWriter-6k' \
    --tensor_model_parallel_size 4 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --train_iters 1000 \
    --eval_iters 50 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 100 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen2.5-7B \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 32768 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --use_flash_attn true
