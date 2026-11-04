#!/bin/bash
# DeepSeek-V4.1 SFT — multi-node (Megatron backend).
#
# 1. Nodes / GPUs: run this SAME script on every node; the scheduler (DLC / slurm / k8s) injects
#      NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT (or export them by hand). NPROC_PER_NODE is the
#      GPU count per node, so world_size = NNODES * NPROC_PER_NODE. To simulate N nodes on ONE machine,
#      launch it N times with a different NODE_RANK and a disjoint CUDA_VISIBLE_DEVICES each, e.g.:
#        NNODES=2 NODE_RANK=0 NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 bash megatron_sft.sh
#        NNODES=2 NODE_RANK=1 NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=2,3 bash megatron_sft.sh
#
# 2. Install (conda env, Python 3.12; SFT needs no vLLM):
#      pip install "git+https://github.com/tastelikefeet/Megatron-LM.git@dsv41-pr7224-engram-local"
#      pip install -e . && pip install -e mcore-bridge                       # ms-swift + V4.1 bridge/loader
#      # transformer_engine must be built for your GPU's real compute capability (the driver value from
#      # torch.cuda.get_device_capability, which can differ from nvidia-smi), else its multi_tensor optimizer
#      # kernels die at optimizer.step() with "no kernel image is available for execution on the device":
#      NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS="90" pip install --no-build-isolation <transformer_engine-src>
#
# 3. No TP/SP: DeepSeek-V4.1 does not support tensor/sequence parallel (we have not implemented it), so
#      tensor_model_parallel_size stays 1 and there is no --sequence_parallel. CP / PP / EP / DP are supported.
set -e
cd "$(git rev-parse --show-toplevel)"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"   # newer libstdc++ (CXXABI) than the system one
export TOKENIZERS_PARALLELISM=false

# ---- distributed launch (scheduler-injected; set NNODES to your cluster -- the Flash defaults below assume 4 nodes x 8 GPUs) ----
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ---- model / data ----
MODEL=${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}
DATASET=${DATASET:-'liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT'}

# ---- parallelism (V4.1: TP fixed 1, no SP; CP/PP/EP/DP supported) ----
TP_SIZE=1
CP_SIZE=${CP_SIZE:-1}
PP_SIZE=${PP_SIZE:-4}
EP_SIZE=${EP_SIZE:-8}

# ---- batch: dp_size = world_size / (TP*CP*PP). global_batch_size must be a multiple of dp_size * micro. ----
WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))
DP_SIZE=$(( WORLD_SIZE / (TP_SIZE * CP_SIZE * PP_SIZE) ))
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}
GLOBAL_BATCH_SIZE=$(( DP_SIZE * MICRO_BATCH_SIZE * GRAD_ACCUM ))

# ---- sequence scale (defaults for Flash; shrink for a small-model smoke test) ----
MAX_LENGTH=${MAX_LENGTH:-8192}
TRAIN_ITERS=${TRAIN_ITERS:-100}
SAVE_STEPS=${SAVE_STEPS:-100}

# Activation recomputation is now Engram-safe (DeepseekV41EngramConfig defers the decision to the V4.1
# backbone, and _checkpointed_forward threads input_ids for the replayed Engram forward): full-layer
# recompute below trades compute for the largest memory saving. For less saving switch to selective with
# engram-safe modules (--recompute_granularity selective --recompute_modules moe moe_act layernorm
# mla_up_proj shared_experts), or set --recompute_granularity none to disable.

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron sft \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --split_dataset_ratio 0.01 \
    --tuner_type full \
    --context_parallel_size ${CP_SIZE} \
    --tensor_model_parallel_size ${TP_SIZE} \
    --expert_model_parallel_size ${EP_SIZE} \
    --expert_tensor_parallel_size 1 \
    --pipeline_model_parallel_size ${PP_SIZE} \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --global_batch_size ${GLOBAL_BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --train_iters ${TRAIN_ITERS} \
    --lr 1e-5 \
    --min_lr 1e-6 \
    --lr_warmup_fraction 0.05 \
    --bf16 true \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer \
    --finetune true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --masked_softmax_fusion false \
    --attention_backend flash \
    --logging_steps 1 \
    --eval_iters 0 \
    --save_steps ${SAVE_STEPS} \
    --no_save_optim true \
    --no_save_rng true \
    --save_safetensors true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --report_to tensorboard \
    --output_dir megatron_output/dsv41-sft
