# 2 * 80GiB
# For inference code, refer to: examples/infer/demo_reranker.py
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-Reranker-8B \
    --task_type generative_reranker \
    --load_safetensors true \
    --save_safetensors true \
    --tuner_type full \
    --dataset MTEB/scidocs-reranking \
    --load_from_cache_file true \
    --split_dataset_ratio 0.02 \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 5e-6 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-7 \
    --max_epochs 1 \
    --save megatron_output/Qwen3-Reranker-8B \
    --save_interval 200 \
    --eval_interval 50 \
    --max_length 4096 \
    --loss_type pointwise_reranker \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4
