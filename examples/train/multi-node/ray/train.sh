# v5 multi-node SFT through Ray: ONE driver command over a pre-started, cluster-wide Ray pool.
#
# Why Ray differs from the torchrun path. There is no per-node command and no NNODES / NODE_RANK /
# MASTER_ADDR: you first bring up a Ray cluster spanning the nodes, then run a single `swift sft
# --mode ray` on any node that can reach it. `--mode ray` makes initialize_twinkle build a 'model'
# DeviceGroup that the Ray workers join, while the driver process only orchestrates and holds NO GPU.
# Ray schedules those workers onto GPUs across the whole cluster, so crossing a node boundary is
# transparent to swift -- the only thing that changes versus single-node Ray is how the cluster was
# started. DO NOT export NPROC_PER_NODE here: that variable makes swift/cli/main.py relaunch under
# torchrun, which is the other (mutually exclusive) launch path. The Ray worker count is the CLI flag
# `--nproc_per_node` instead.
#
# 1) start the cluster (once per node). Head node, then every worker node points at it:
#      head:    ray start --head --port=6379 --num-gpus=2
#      worker:  ray start --address=<head_ip>:6379 --num-gpus=2
# 2) run the driver on the head (RAY_ADDRESS finds the cluster; --nproc_per_node is the TOTAL rank
#    count across ALL nodes, i.e. nodes x gpus-per-node = 2 x 2 = 4 here):
#      RAY_ADDRESS=<head_ip>:6379 bash train.sh
#
# SINGLE-BOX simulation (how this was smoke-validated): one node stands in for the cluster, so the
# multi-worker Ray placement is exercised for real while the workers happen to share one machine.
#   ray start --head --num-gpus=4     # or let ray.init() auto-start a local cluster
#   RAY_ADDRESS=auto NPROC_TOTAL=4 CUDA_VISIBLE_DEVICES=1,2,3,4 bash train.sh
# Truly spanning physical machines only needs the workers' `ray start --address` to dial the same head.
#
# Batch semantics under Ray MATCH torchrun: per_device_train_batch_size is PER DP RANK, not a global
# batch. The driver's DataLoader builds the GLOBAL batch (per_device x dp = 8 x 4 = 32 here, sized by
# _twinkle_loader_layout) and forward_backward(dispatch='slice_dp') scatters it back to per_device rows
# on each rank, so the global batch is per_device x dp x gradient_accumulation_steps exactly as under
# torchrun. The transformers backend has no TP/PP sharding, so the Ray mesh is pure data parallelism.
# (With 500 rows / global batch 32 / ga 2, one epoch is ~7 optimizer steps -- a short run by design.)
#
# --attn_impl flash_attention_2 is pinned: on the validation box (torch 2.11 + cuDNN 9.19) the default
# SDPA path picks a cuDNN attention kernel whose bf16 backward returns NaN on longer rows -- clean
# through ~step 25, then grad_norm goes non-finite and loss becomes nan. FA2 avoids it (and is standard
# for multi-node long-sequence training); `--attn_impl eager` is the fallback where flash-attn is absent.
RAY_ADDRESS=${RAY_ADDRESS:-auto} \
NPROC_TOTAL=${NPROC_TOTAL:-4} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --mode ray \
    --nproc_per_node "$NPROC_TOTAL" \
    --ray_exp_name qwen05b_multinode_ray \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 2 \
    --output_dir output \
    --report_to tensorboard
