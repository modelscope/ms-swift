# v5 Megatron-backend MoE SFT: expert parallelism (ep4) on 4 GPUs, Qwen3-30B-A3B (128 experts, top-8, 3B active).
#
# Same launcher as the dense tp_pp example -- `swift sft --backend megatron`, NPROC_PER_NODE relaunches under
# torch.distributed.run -- the only thing that changes is the parallel axis. tp_pp splits every weight matrix
# (tensor parallel) and the layer stack (pipeline parallel); an MoE instead splits the EXPERTS: with
# --expert_model_parallel_size 4 the 128 experts are sharded 4-way (32 per rank) and each token is routed to
# its top-8 experts across ranks via an all-to-all (--moe_token_dispatcher_type alltoall). The attention/dense
# part is replicated, so ep (not tp) is the axis that makes a sparse model fit. The world size must equal
# tp x pp x ep x dp; here it is ep4 x dp1 = 4 = NPROC_PER_NODE.
#
# MoE-only knobs (all MegatronConfig/MoEConfig, read only on this backend): --moe_grouped_gemm runs the 32
# local experts as one grouped GEMM instead of a loop; --moe_permute_fusion folds the token-permute into the
# dispatch kernel; --moe_aux_loss_coeff weights the load-balancing loss that keeps the router from collapsing
# onto a few experts. Qwen3-30B-A3B has NO shared expert (num_shared_experts is unset), so
# --moe_shared_expert_overlap is deliberately omitted -- it would be a dead flag on this model.
#
# Megatron batch knobs, not per_device_train_batch_size: --micro_batch_size is per rank per step, and
# --global_batch_size is the whole-optimizer-step batch across all ranks (must be a multiple of
# micro_batch_size x dp). Megatron derives gradient accumulation from the two.
#
# DATASET CHOICE = VISIBLE CONVERGENCE. `swift/self-cognition` is new knowledge the model must fit, so the
# loss falls clearly; an instruct model on alpaca plateaus and looks flat. Keep --num_train_epochs 3.
#
# SCALING UP. Bigger sparse models add pipeline parallel and more expert parallel across more nodes, e.g.
# Qwen3-235B-A22B or DeepSeek-V3: raise --expert_model_parallel_size (and --pipeline_model_parallel_size),
# add NNODES/NODE_RANK/MASTER_ADDR exactly as in examples/train/multi-node/torchrun, and switch --tuner lora
# to full-parameter training with --use_distributed_optimizer. When the optimizer state no longer fits,
# offload it to CPU with --optimizer_cpu_offload true --optimizer_offload_fraction 1. Resources: the ep4 x
# 4-GPU layout here fits Qwen3-30B-A3B with LoRA + full recompute on ~40GB-class GPUs; a 235B/671B MoE needs
# a multi-node cluster.
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
USE_SWIFT_V5=1 \
swift sft \
    --backend megatron \
    --model Qwen/Qwen3-30B-A3B \
    --dataset 'swift/self-cognition#600' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --expert_model_parallel_size 4 \
    --moe_token_dispatcher_type alltoall \
    --moe_grouped_gemm true \
    --moe_permute_fusion true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --attention_backend flash \
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
