PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-0.6B \
    --load_safetensors true \
    --save_safetensors true \
    --dataset 'swift_shuf_19k_data.jsonl' \
    --tensor_model_parallel_size 1 \
    --sequence_parallel true \
    --micro_batch_size 4 \
    --global_batch_size 64 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --max_epochs 1 \
    --save megatron_output/Qwen3-0.6B/eaft \
    --save_interval 100 \
    --max_length 16384 \
    --system 'You are a helpful assistant.' \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --tensorboard_dir /tensorboard/Qwen3-0.6B/eaft \
    --enable_eaft_loss true \
    --eaft_alpha 1.0

