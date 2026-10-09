# v5 Megatron-backend SFT with tensor parallelism (tp2) on 2 GPUs.
#
# v5 has NO separate `megatron` command: twinkle unifies both stacks, so the Megatron path is selected by
# `--backend megatron` on `swift sft`. NPROC_PER_NODE relaunches under torch.distributed.run, and the
# Megatron mesh is built from the parallel-size flags. Megatron uses its own batch knobs
# (--micro_batch_size / --global_batch_size) rather than per_device_train_batch_size, and
# --sequence_parallel here is Megatron's TP-region sequence parallelism (distinct from the transformers
# Ulysses --sequence_parallel_size). The 0.5B model keeps the example light and locally smoke-testable.
#
# DATASET CHOICE = VISIBLE CONVERGENCE. `swift/self-cognition` (not alpaca-gpt4): an already-instruct
# Qwen2.5-0.5B plateaus at ~1.6 loss on alpaca (no headroom), so the run looks flat; self-cognition is new
# knowledge it must fit, so the loss falls clearly (measured 2.6 -> 0.28 on the transformers path). Keep
# --num_train_epochs 3 so the descent completes.
#
# PIPELINE PARALLEL: add `--pipeline_model_parallel_size 2` (and NPROC_PER_NODE=4 / CUDA_VISIBLE_DEVICES=0,1,2,3
# for tp2 x pp2), or state the whole layout at once with `--parallel_spec tp2pp2`. Megatron supports only
# causal_lm SFT -- seq_cls/embedding/reranker stay on the transformers backend (drop --backend).
#
# SCALING UP. For a larger dense model (7B/32B/72B) raise --tensor_model_parallel_size, add
# --pipeline_model_parallel_size across more GPUs/nodes (NNODES/NODE_RANK/MASTER_ADDR exactly as in
# examples/train/multi-node/torchrun), and switch --tuner lora to full-parameter training with
# --use_distributed_optimizer. When the optimizer state no longer fits, offload it to CPU with
# --optimizer_cpu_offload true --optimizer_offload_fraction 1 (slower steps, far less GPU memory).
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
USE_SWIFT_V5=1 \
swift sft \
    --backend megatron \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset 'swift/self-cognition#600' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --num_train_epochs 3 \
    --max_length 2048 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_safetensors true \
    --output_dir output \
    --report_to tensorboard
