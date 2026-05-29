# 2 * 80GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron sft \
    --model /metax0402/metax0402/m01119/models/Qwen/Qwen2.5-7B \
    --save_safetensors true \
    --dataset '/metax0402/metax0402/m01119/datasets/AI-ModelScope/alpaca-gpt4-data-zh#1500' \
    --tensor_model_parallel_size 4 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --num_train_epochs 1 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct \
    --save_steps 100 \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --model_author swift \
    --model_name swift-robot \
    --enable_profiler true \
    --profiler_save_path ./profiler_output \
    --profiler_ranks 0 1\
    --profiler_contents "cpu" "cuda" "stack" \
    --profiler_steps 1  \
