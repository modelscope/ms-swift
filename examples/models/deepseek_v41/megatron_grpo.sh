#!/bin/bash
# DeepSeek-V4.1 GRPO — multi-node (Megatron + vLLM colocate).
#
# 1. Nodes / GPUs: run this SAME script on every node; the scheduler (DLC / slurm / k8s) injects
#      NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT (or export them by hand). NPROC_PER_NODE is the
#      GPU count per node, so world_size = NNODES * NPROC_PER_NODE. To simulate N nodes on ONE machine,
#      launch it N times with a different NODE_RANK and a disjoint CUDA_VISIBLE_DEVICES each, e.g.:
#        NNODES=2 NODE_RANK=0 NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 bash megatron_grpo.sh
#        NNODES=2 NODE_RANK=1 NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=2,3 bash megatron_grpo.sh
#
# 2. Install (conda env, Python 3.12):
#      # rollout needs a vLLM build with native DeepSeek-V4.1 support -- use the official image (no pip
#      # wheel yet): vllm/vllm-openai:deepseekv41-flash
#      pip install "git+https://github.com/tastelikefeet/Megatron-LM.git@dsv41-pr7224-engram-local"
#      pip install -e . && pip install -e mcore-bridge                       # ms-swift + V4.1 bridge/loader
#      # transformer_engine must be built for your GPU's real compute capability (the driver value from
#      # torch.cuda.get_device_capability, which can differ from nvidia-smi), else its multi_tensor optimizer
#      # kernels die at optimizer.step() with "no kernel image is available for execution on the device":
#      NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS="90" pip install --no-build-isolation <transformer_engine-src>
#
# 3. Only if you build vLLM from main instead of the official image: main reads candidate_source_layer_id,
#      which is null when CSA2 candidates are off, so `>= 0` raises TypeError. After each
#      getattr(config, "candidate_source_layer_id", -1) add `if x is None: x = -1`, in
#      vllm/models/deepseek_v41/{attention.py, nvidia/model.py, amd/model.py}.
#
# 4. No TP/SP: DeepSeek-V4.1 does not support tensor/sequence parallel (we have not implemented it), so
#      tensor_model_parallel_size stays 1 and there is no --sequence_parallel. CP / PP / EP / DP are supported.
#      --vllm_tensor_parallel_size is the inference-side TP, independent of the (unsupported) training TP;
#      keep it within one node.
set -e
cd "$(git rev-parse --show-toplevel)"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"   # newer libstdc++ (CXXABI) than the system one
export VLLM_USE_DEEP_GEMM=0                                          # fp8 GEMM ext, unused for this bf16 run
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
DATASET=${DATASET:-'open-r1/DAPO-Math-17k-Processed'}

# ---- parallelism (V4.1: TP fixed 1, no SP; CP/PP/EP/DP supported) ----
TP_SIZE=1
CP_SIZE=${CP_SIZE:-1}
PP_SIZE=${PP_SIZE:-4}
EP_SIZE=${EP_SIZE:-8}
VLLM_TP=${VLLM_TP:-8}

# ---- batch: dp_size = world_size / (TP*CP*PP). Keep 1 rollout prompt per rank, i.e.
#      num_rollout_prompt = global_batch_size * steps_per_generation / num_generations = dp_size. ----
WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))
DP_SIZE=$(( WORLD_SIZE / (TP_SIZE * CP_SIZE * PP_SIZE) ))
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
STEPS_PER_GENERATION=${STEPS_PER_GENERATION:-1}
GLOBAL_BATCH_SIZE=$(( DP_SIZE * NUM_GENERATIONS * STEPS_PER_GENERATION ))
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}

# ---- sequence / rollout scale (defaults for Flash; shrink for a small-model smoke test) ----
MAX_LENGTH=${MAX_LENGTH:-8192}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-8192}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.4}
TRAIN_ITERS=${TRAIN_ITERS:-100}
SAVE_STEPS=${SAVE_STEPS:-100}

# Activation recomputation is now Engram-safe (DeepseekV41EngramConfig defers the decision to the V4.1
# backbone, and _checkpointed_forward threads input_ids for the replayed Engram forward): full-layer
# recompute below trades compute for the largest memory saving. For less saving switch to selective with
# engram-safe modules (--recompute_granularity selective --recompute_modules moe moe_act layernorm
# mla_up_proj shared_experts), or set --recompute_granularity none to disable.
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type grpo \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --split_dataset_ratio 0 \
    --tuner_type full \
    --context_parallel_size ${CP_SIZE} \
    --tensor_model_parallel_size ${TP_SIZE} \
    --expert_model_parallel_size ${EP_SIZE} \
    --expert_tensor_parallel_size 1 \
    --pipeline_model_parallel_size ${PP_SIZE} \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --num_generations ${NUM_GENERATIONS} \
    --steps_per_generation ${STEPS_PER_GENERATION} \
    --global_batch_size ${GLOBAL_BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --max_completion_length ${MAX_COMPLETION_LENGTH} \
    --train_iters ${TRAIN_ITERS} \
    --reward_funcs accuracy format \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization ${VLLM_GPU_UTIL} \
    --vllm_tensor_parallel_size ${VLLM_TP} \
    --vllm_max_model_len ${VLLM_MAX_MODEL_LEN} \
    --vllm_enforce_eager true \
    --lr 1e-6 \
    --min_lr 1e-7 \
    --lr_warmup_fraction 0.05 \
    --bf16 true \
    --beta 0.00 \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --importance_sampling_level sequence \
    --temperature 1.0 \
    --sleep_level 2 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer true \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer \
    --finetune true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --masked_softmax_fusion false \
    --attention_backend flash \
    --padding_free true \
    --logging_steps 1 \
    --eval_iters 0 \
    --save_steps ${SAVE_STEPS} \
    --no_save_optim true \
    --no_save_rng true \
    --save_safetensors true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --log_completions true \
    --report_to tensorboard \
    --output_dir megatron_output/dsv41-grpo
