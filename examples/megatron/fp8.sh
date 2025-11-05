# bf16: 4 * 56GiB; 17.5s/it
# fp8: 4 * 58GiB; 12.5s/it
# If you encounter a cuBLAS error, please install transformer_engine using the method below.
# pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.5#egg=transformer_engine[pytorch]
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron sft \
    --load Qwen2.5-7B-mcore \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 2 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 3 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen2.5-7B \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --fp8_format hybrid \
    --fp8_param_gather true \
    --sequence_parallel true \
    --attention_backend flash
