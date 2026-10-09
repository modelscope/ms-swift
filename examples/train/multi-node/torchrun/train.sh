# v5 multi-node SFT under torchrun: run THIS SAME command on every node, only NODE_RANK differs.
#
# Launch model. Unlike the legacy form (hand-writing `torchrun ... swift/cli/sft.py`), v5 drives the
# launcher through environment variables: setting NNODES (or NPROC_PER_NODE) makes swift/cli/main.py
# relaunch the dev CLI under `torch.distributed.run`, forwarding NPROC_PER_NODE / NNODES / NODE_RANK /
# MASTER_ADDR / MASTER_PORT verbatim (get_torchrun_args). So a node only exports the five variables and
# calls `swift sft`; there is no torchrun line to maintain per node.
#
# Topology. World size = NNODES x NPROC_PER_NODE. Each rank runs the same recipe over its shard
# (mode='local': twinkle.initialize builds a pure data-parallel mesh from Platform.get_world_size(),
# which reads the WORLD_SIZE torchrun exports). The transformers backend has no TP/PP weight sharding,
# so every rank holds a full replica and only the data is split. per_device_train_batch_size is PER
# RANK; the global batch is per_device x world x gradient_accumulation_steps.
#
# REAL multi-node (2 nodes x 2 GPUs). Run on each node, changing only NODE_RANK and the local GPUs;
# MASTER_ADDR is the rank-0 node's reachable IP on BOTH nodes, and all nodes must agree on NNODES,
# NPROC_PER_NODE and MASTER_PORT:
#   node 0:  NODE_RANK=0 MASTER_ADDR=<rank0_ip> CUDA_VISIBLE_DEVICES=0,1 bash train.sh
#   node 1:  NODE_RANK=1 MASTER_ADDR=<rank0_ip> CUDA_VISIBLE_DEVICES=0,1 bash train.sh
#
# SINGLE-BOX simulation (how this was smoke-validated): two terminals on one machine, MASTER_ADDR over
# loopback, disjoint GPUs so the 2 nodes do not collide, NPROC_PER_NODE=2 -> world=4. The rendezvous
# and the cross-"node" all-reduce are the real thing; only the network is loopback.
#   terminal A:  NODE_RANK=0 MASTER_ADDR=127.0.0.1 CUDA_VISIBLE_DEVICES=1,2 NPROC_PER_NODE=2 bash train.sh
#   terminal B:  NODE_RANK=1 MASTER_ADDR=127.0.0.1 CUDA_VISIBLE_DEVICES=3,4 NPROC_PER_NODE=2 bash train.sh
#
# Full-parameter multi-node training usually also shards the optimizer/parameters to fit: add
# `--deepspeed zero2` (or zero3) -- it is forwarded to the same launcher and needs no other change.
#
# --attn_impl flash_attention_2 is pinned: on the validation box (torch 2.11 + cuDNN 9.19) the default
# SDPA path picks a cuDNN attention kernel whose bf16 backward returns NaN on longer rows -- clean
# through ~step 25, then grad_norm goes non-finite and loss becomes nan. FA2 avoids it (and is standard
# for multi-node long-sequence training); `--attn_impl eager` is the fallback where flash-attn is absent.
NNODES=${NNODES:-2} \
NPROC_PER_NODE=${NPROC_PER_NODE:-2} \
NODE_RANK=${NODE_RANK:?set NODE_RANK: 0 on the rank-0 node, 1..NNODES-1 on the workers} \
MASTER_ADDR=${MASTER_ADDR:?set MASTER_ADDR to the rank-0 node IP (127.0.0.1 for a single-box sim)} \
MASTER_PORT=${MASTER_PORT:-29500} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
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
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
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
