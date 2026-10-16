# Requires megatron-core>=0.18. Use MLP_CHUNKS_FOR_TRAINING=1 for the baseline.
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
megatron sft \
    --model Qwen/Qwen3-0.6B \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-en#1000' \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --mlp_chunks_for_training "${MLP_CHUNKS_FOR_TRAINING:-4}" \
    --recompute_granularity selective \
    --recompute_modules core_attn \
    --tensor_model_parallel_size 1 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --train_iters 20 \
    --max_length 4096 \
    --lr 1e-5 \
    --logging_steps 1 \
    --output_dir "megatron_output/chunked_mlp_${MLP_CHUNKS_FOR_TRAINING:-4}" \
    --no_save_optim true \
    --no_save_rng true
