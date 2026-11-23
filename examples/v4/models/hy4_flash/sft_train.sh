#!/bin/bash
export USE_SWIFT_V5=0
# ============================================================================
# ms-swift LLM full fine-tuning + DeepSpeed ZeRO-3 training launch script
#
# Source: Tencent-Hunyuan/Hy4-flash/finetune/ms_swift_support/sft_train.sh
#
# This script uses pdsh to automatically distribute to all nodes.
# You only need to run it once on any node.
#
# Usage:
#   nohup bash sft_train.sh > swift_llm_full.log 2>&1 &
# ============================================================================

set -euo pipefail

# -------------------- Network Configuration --------------------
NET_TYPE="high"
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_TIMEOUT=24
export NCCL_NVLS_ENABLE=0
export NCCL_MPI_PROFILE_PRIMS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
fi

export DISABLE_VERSION_CHECK=1

# -------------------- Node Configuration --------------------
export HOST_GPU_NUM=8
# IP list, comma separated. e.g. "10.0.0.1,10.0.0.2" or single node "127.0.0.1"
export IP_LIST=${IP_LIST:-"127.0.0.1"}

MASTER_PORT=${MASTER_PORT:-29500}

IFS=',' read -ra IP_ARRAY <<< "$IP_LIST"
NODES=${#IP_ARRAY[@]}
MASTER_ADDR=${IP_ARRAY[0]}

# -------------------- Paths --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_FILE="${SCRIPT_DIR}/hy_v4_flash_swift_patches.py"
# Local directory produced by
# `modelscope download Tencent-Hunyuan/Hy4-flash --local-dir Tencent-Hunyuan/Hy4-flash`
HY4_CKPT_DIR=${HY4_CKPT_DIR:-"./Tencent-Hunyuan/Hy4-flash"}

# -------------------- Multi-node: pdsh auto-dispatch --------------------
if [ ${NODES} -gt 1 ] && [ -z "${_WORKER_MODE:-}" ]; then
    echo "============================================"
    echo "  HYV4-Flash ms-swift LLM SFT Training (Full + DeepSpeed ZeRO-3)"
    echo "  Launcher mode: distributing to ${NODES} nodes via pdsh"
    echo "  Nodes: ${IP_LIST}"
    echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
    echo "  GPUs per node: ${HOST_GPU_NUM}"
    echo "  Total GPUs: $((NODES * HOST_GPU_NUM))"
    echo "============================================"

    PDSH_HOSTS=$(echo "${IP_LIST}" | tr ',' ',')

    pdsh -S -f 1024 -w "${PDSH_HOSTS}" \
        "export NCCL_DEBUG=WARN; \
         export NCCL_P2P_LEVEL=NVL; \
         export NCCL_IB_TIMEOUT=24; \
         export NCCL_NVLS_ENABLE=0; \
         export NCCL_MPI_PROFILE_PRIMS_ENABLE=0; \
         export CUDA_DEVICE_MAX_CONNECTIONS=1; \
         export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600; \
         export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128; \
         export NCCL_IB_GID_INDEX=3; \
         export NCCL_IB_SL=3; \
         export NCCL_CHECK_DISABLE=1; \
         export NCCL_P2P_DISABLE=0; \
         export NCCL_IB_DISABLE=0; \
         export NCCL_LL_THRESHOLD=16384; \
         export NCCL_IB_CUDA_SUPPORT=1; \
         export NCCL_SOCKET_IFNAME=bond1; \
         export UCX_NET_DEVICES=bond1; \
         export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6; \
         export NCCL_COLLNET_ENABLE=0; \
         export SHARP_COLL_ENABLE_SAT=0; \
         export NCCL_NET_GDR_LEVEL=2; \
         export NCCL_IB_QPS_PER_CONNECTION=4; \
         export NCCL_IB_TC=160; \
         export NCCL_PXN_DISABLE=1; \
         export DISABLE_VERSION_CHECK=1; \
         export HY4_CKPT_DIR='${HY4_CKPT_DIR}'; \
         export IP_LIST='${IP_LIST}'; \
         export _WORKER_MODE=1; \
         cd ${SCRIPT_DIR}; \
         bash ${SCRIPT_DIR}/sft_train.sh"
    exit $?
fi

# -------------------- Worker Mode: determine node rank and launch --------------------
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export NNODES="${NODES}"

if [ ${NODES} -gt 1 ]; then
    LOCAL_IP=$(hostname -i | awk '{print $1}')
    NODE_RANK=0
    for i in "${!IP_ARRAY[@]}"; do
        if [[ "${IP_ARRAY[$i]}" == "${LOCAL_IP}" ]]; then
            NODE_RANK=$i
            break
        fi
    done
    export RANK="${NODE_RANK}"
else
    export RANK=0
fi

echo "============================================"
echo "  HYV4-Flash ms-swift LLM SFT Training"
echo "  Worker mode: Node ${RANK}/${NNODES}"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  GPUs per node: ${HOST_GPU_NUM}"
echo "  Total GPUs: $((NODES * HOST_GPU_NUM))"
echo "============================================"

# -------------------- Launch --------------------
# Stagger node launches to avoid cephfs I/O congestion
if [ "${RANK}" -gt 0 ]; then
    echo "Node ${RANK}: staggering launch by $((RANK * 5)) seconds to reduce cephfs I/O pressure..."
    sleep $((RANK * 5))
fi

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${SCRIPT_DIR}"

# DeepSpeed config path (absolute)
DS_CONFIG="${HY4_CKPT_DIR}/finetune/deepspeed_support/ds_zero3_offload.json"

# NOTE: --config is NOT supported by the current transformers HfArgumentParser
# when using torchrun -m swift.cli.sft. All parameters passed as CLI args.
torchrun \
    --nnodes "${NNODES}" \
    --node_rank "${RANK}" \
    --nproc_per_node "${HOST_GPU_NUM}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    -m swift.cli.sft \
    --custom_register_path "${PATCHES_FILE}" \
    --model "${HY4_CKPT_DIR}" \
    --model_type hy_v4_flash_vl \
    --template hy_v4_flash \
    --torch_dtype bfloat16 \
    --attn_impl eager \
    --tuner_type full \
    --dataset "${HY4_CKPT_DIR}/finetune/data/example_data.jsonl" \
--max_length 1024 \
    --lazy_tokenize true \
    --dataset_num_proc 4 \
    --output_dir saves/hy_v4_flash/full/sft_ds \
    --save_steps 20 \
    --save_strategy steps \
    --save_total_limit 3 \
    --logging_steps 1 \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --max_steps 20 \
    --num_train_epochs 3.0 \
    --warmup_steps 0 \
    --lr_scheduler_type cosine \
    --bf16 true \
    --deepspeed "${DS_CONFIG}" \
    --optim adamw_torch \
    --use_logits_to_keep false \
    --gradient_checkpointing true \
    --max_grad_norm 0.0 \
    --ddp_timeout 180000000 \
    --seed 42 \
    --ignore_data_skip true
