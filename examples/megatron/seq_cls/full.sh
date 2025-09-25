# 4 * 60GiB; 7.5s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron sft \
    --load Qwen2.5-VL-7B-Instruct-mcore \
    --dataset 'tany0699/garbage265#20000' \
    --load_from_cache_file true \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --packing true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --split_dataset_ratio 0.01 \
    --micro_batch_size 2 \
    --global_batch_size 4 \
    --model_kwargs '{"max_pixels": 1003520}' \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --max_epochs 1 \
    --save megatron_output/Qwen2.5-VL-7B-Instruct \
    --save_interval 200 \
    --eval_interval 200 \
    --vit_gradient_checkpointing true \
    --max_length 8192 \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 32 \
    --num_labels 265 \
    --task_type seq_cls \
