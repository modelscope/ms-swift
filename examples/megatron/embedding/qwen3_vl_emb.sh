# 2 * 80GiB
# For inference code, refer to: examples/infer/demo_embedding.py
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
INFONCE_TEMPERATURE=0.1 \
megatron sft \
    --model Qwen/Qwen3-VL-Embedding-8B \
    --task_type embedding \
    --save_safetensors true \
    --tuner_type full \
    --dataset swift/TextCaps:emb \
    --load_from_cache_file true \
    --split_dataset_ratio 0.02 \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 16 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 5e-6 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-7 \
    --num_train_epochs 1 \
    --output_dir megatron_output/Qwen3-VL-Embedding-8B \
    --save_interval 200 \
    --eval_interval 100 \
    --max_length 8192 \
    --loss_type infonce \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4
