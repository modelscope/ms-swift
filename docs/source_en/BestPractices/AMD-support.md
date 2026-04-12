# AMD GPU Support

## 1. Environment setup

### 1.1 Base environment

Pull the ms-swift image built for the AMD ROCm stack, then start the container with the commands below.

If you need a newer ms-swift version, upgrade with pip or install from source code (adding `--no-deps` is recommended to avoid pulling in dependency upgrades that may cause issues).

```bash
IMAGE_NAME=amdagi/modelscope:ubuntu22.04-rocm7.2.0-py312-torch2.10.0-vllm0.18.1-modelscope1.35.1-swift4.1.0
docker pull ${IMAGE_NAME}

CONTAINER_NAME=swift_test
docker run -it --network=host --ipc=host --privileged --group-add video \
    --device=/dev/dri --device=/dev/kfd \
    --shm-size 512G --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
    --name ${CONTAINER_NAME} \
    ${IMAGE_NAME} \
    /bin/bash
```

### 1.2 Environment check

- Confirm the availability of AMD devices for PyTorch in the container.

```bash
python -c "import torch;print(torch.cuda.is_available())"  # output: True
```

- Inspect GPU topology and NUMA: `rocm-smi --showtopo`

```
============================ ROCm System Management Interface ============================
WARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status

================================ Weight between two GPUs =================================
       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7
GPU0   0            15           15           15           15           15           15           15
GPU1   15           0            15           15           15           15           15           15
GPU2   15           15           0            15           15           15           15           15
GPU3   15           15           15           0            15           15           15           15
GPU4   15           15           15           15           0            15           15           15
GPU5   15           15           15           15           15           0            15           15
GPU6   15           15           15           15           15           15           0            15
GPU7   15           15           15           15           15           15           15           0

================================= Hops between two GPUs ==================================
       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7
GPU0   0            1            1            1            1            1            1            1
GPU1   1            0            1            1            1            1            1            1
GPU2   1            1            0            1            1            1            1            1
GPU3   1            1            1            0            1            1            1            1
GPU4   1            1            1            1            0            1            1            1
GPU5   1            1            1            1            1            0            1            1
GPU6   1            1            1            1            1            1            0            1
GPU7   1            1            1            1            1            1            1            0

=============================== Link Type between two GPUs ===============================
       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7
GPU0   0            XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI
GPU1   XGMI         0            XGMI         XGMI         XGMI         XGMI         XGMI         XGMI
GPU2   XGMI         XGMI         0            XGMI         XGMI         XGMI         XGMI         XGMI
GPU3   XGMI         XGMI         XGMI         0            XGMI         XGMI         XGMI         XGMI
GPU4   XGMI         XGMI         XGMI         XGMI         0            XGMI         XGMI         XGMI
GPU5   XGMI         XGMI         XGMI         XGMI         XGMI         0            XGMI         XGMI
GPU6   XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         0            XGMI
GPU7   XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         0

======================================= Numa Nodes =======================================
GPU[0]          : (Topology) Numa Node: 0
GPU[0]          : (Topology) Numa Affinity: 0
GPU[1]          : (Topology) Numa Node: 0
GPU[1]          : (Topology) Numa Affinity: 0
GPU[2]          : (Topology) Numa Node: 0
GPU[2]          : (Topology) Numa Affinity: 0
GPU[3]          : (Topology) Numa Node: 0
GPU[3]          : (Topology) Numa Affinity: 0
GPU[4]          : (Topology) Numa Node: 1
GPU[4]          : (Topology) Numa Affinity: 1
GPU[5]          : (Topology) Numa Node: 1
GPU[5]          : (Topology) Numa Affinity: 1
GPU[6]          : (Topology) Numa Node: 1
GPU[6]          : (Topology) Numa Affinity: 1
GPU[7]          : (Topology) Numa Node: 1
GPU[7]          : (Topology) Numa Affinity: 1
================================== End of ROCm SMI Log ===================================
```

- Check GPU utilization and VRAM usage (`rocm-smi` or `rocm-smi -u --showmeminfo vram`):

```
# output of 'rocm-smi'
============================================ ROCm System Management Interface ============================================
====================================================== Concise Info ======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)
==========================================================================================================================
0       2     0x74a2,   1017   43.0°C      155.0W    NPS1, SPX, 0        94Mhz   900Mhz  0%   auto  650.0W  0%     0%
1       3     0x74a2,   47713  41.0°C      155.0W    NPS1, SPX, 0        91Mhz   900Mhz  0%   auto  650.0W  0%     0%
2       4     0x74a2,   37449  45.0°C      159.0W    NPS1, SPX, 0        95Mhz   900Mhz  0%   auto  650.0W  0%     0%
3       5     0x74a2,   11217  41.0°C      155.0W    NPS1, SPX, 0        95Mhz   900Mhz  0%   auto  650.0W  0%     0%
4       6     0x74a2,   41880  44.0°C      160.0W    NPS1, SPX, 0        91Mhz   900Mhz  0%   auto  650.0W  0%     0%
5       7     0x74a2,   6656   42.0°C      157.0W    NPS1, SPX, 0        95Mhz   900Mhz  0%   auto  650.0W  0%     0%
6       8     0x74a2,   12840  45.0°C      160.0W    NPS1, SPX, 0        96Mhz   900Mhz  0%   auto  650.0W  0%     0%
7       9     0x74a2,   35760  43.0°C      158.0W    NPS1, SPX, 0        107Mhz  900Mhz  0%   auto  650.0W  0%     0%
==========================================================================================================================
================================================== End of ROCm SMI Log ===================================================

# output of 'rocm-smi -u --showmeminfo vram'
============================ ROCm System Management Interface ============================
=================================== % time GPU is busy ===================================
GPU[0]          : GPU use (%): 0
GPU[0]          : GFX Activity: 3862538534
GPU[1]          : GPU use (%): 0
GPU[1]          : GFX Activity: 4053246251
GPU[2]          : GPU use (%): 0
GPU[2]          : GFX Activity: 3114103535
GPU[3]          : GPU use (%): 0
GPU[3]          : GFX Activity: 4026776444
GPU[4]          : GPU use (%): 0
GPU[4]          : GFX Activity: 1224255679
GPU[5]          : GPU use (%): 0
GPU[5]          : GFX Activity: 1191191242
GPU[6]          : GPU use (%): 0
GPU[6]          : GFX Activity: 1184652679
GPU[7]          : GPU use (%): 0
GPU[7]          : GFX Activity: 2145209382
==========================================================================================
================================== Memory Usage (Bytes) ==================================
GPU[0]          : VRAM Total Memory (B): 206141652992
GPU[0]          : VRAM Total Used Memory (B): 297611264
GPU[1]          : VRAM Total Memory (B): 206141652992
GPU[1]          : VRAM Total Used Memory (B): 297623552
GPU[2]          : VRAM Total Memory (B): 206141652992
GPU[2]          : VRAM Total Used Memory (B): 297623552
GPU[3]          : VRAM Total Memory (B): 206141652992
GPU[3]          : VRAM Total Used Memory (B): 297623552
GPU[4]          : VRAM Total Memory (B): 206141652992
GPU[4]          : VRAM Total Used Memory (B): 297623552
GPU[5]          : VRAM Total Memory (B): 206141652992
GPU[5]          : VRAM Total Used Memory (B): 297623552
GPU[6]          : VRAM Total Memory (B): 206141652992
GPU[6]          : VRAM Total Used Memory (B): 297623552
GPU[7]          : VRAM Total Memory (B): 206141652992
GPU[7]          : VRAM Total Used Memory (B): 297623552
==========================================================================================
================================== End of ROCm SMI Log ===================================
```

## 2. Run examples

### 2.1 Full fine-tuning Qwen3.5 with Megatron-Swift

AMD GPUs often have large VRAM, so you can tune several knobs together to improve training throughput:

- **Parallelism tuning**: Large per-GPU memory lets you reduce communication from aggressive splits (prefer tuning PP/EP before TP).
- **Optimizer CPU offload**: If VRAM allows, disable with `--optimizer_cpu_offload false`.
- **Activation / gradient checkpointing**: If VRAM allows, use `--recompute_granularity none`, or `--recompute_granularity selective` with `--recompute_modules` for finer control.
- **MoE models**: Set `export NVTE_USE_CUTLASS_GROUPED_GEMM=1` for the optimized grouped GEMM kernel.
- **Models with GatedDeltaNet**: Set `SWIFT_USE_MCORE_GDN=1` to use the Megatron-Core implementation.
- **Stability on some AMD GPUs**: Set `export HSA_NO_SCRATCH_RECLAIM=1` to avoid known issues and stabilize performance.

Single-node training:

```bash
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_USE_CUTLASS_GROUPED_GEMM=1

output_dir=${PWD}/megatron_output/Qwen3.5-35B-A3B
mkdir -p ${output_dir}
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
log_file=${output_dir}/"1node_full_megatron_Qwen3.5-35B-A3B_${current_time}.log"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
SKIP_MULTIMODAL_MTP_VALIDATION=1 \
SWIFT_USE_MCORE_GDN=1 \
megatron sft \
    --model Qwen/Qwen3.5-35B-A3B \
    --dataset 'AI-ModelScope/LongAlpaca-12k' \
    --save_safetensors true \
    --load_from_cache_file true \
    --tuner_type full \
    --add_non_thinking_prefix true \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --sequence_parallel true \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --moe_expert_capacity_factor 2 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --recompute_granularity selective \
    --recompute_modules core_attn mlp moe \
    --num_train_epochs 500 \
    --group_by_length true \
    --finetune true \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --max_length 16384 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --optimizer_cpu_offload false \
    --attention_backend flash \
    --padding_free false \
    --output_dir ${output_dir} \
    2>&1 | tee ${log_file}
```

Multi-node training:

```bash
export NNODES=2  # example: 2 nodes
export NODE_RANK=0  # 0 on master, 1 on workers
export MASTER_ADDR=<MASTER_NODE_IP>  # set to master node IP
export MASTER_PORT=29500  # communication port
export NCCL_SOCKET_IFNAME=ens50f1np1  # actual NIC name, check with ifconfig
export GLOO_SOCKET_IFNAME=ens50f1np1  # same as above
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3  # IB HCAs, check with ibv_devices
export NCCL_IB_GID_INDEX=3

# Main training script below: same as single-node script above
...
```

### 2.2 Reinforcement learning training for Qwen3.5 with Megatron-Swift

```bash
# Single-node training example
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_USE_CUTLASS_GROUPED_GEMM=1

SYSTEM_PROMPT="""You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."""

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-35B-A3B \
    --save_safetensors true \
    --enable_thinking false \
    --merge_lora true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --pipeline_model_parallel_size 1 \
    --moe_permute_fusion true \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --system "$SYSTEM_PROMPT" \
    --num_train_epochs 1 \
    --global_batch_size 64 \
    --micro_batch_size 1 \
    --steps_per_generation 2 \
    --num_generations 8 \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 9192 \
    --max_length 1000 \
    --max_completion_length 8192 \
    --tuner_type lora \
    --target_modules all-linear \
    --lr 5e-5 \
    --bf16 true \
    --beta 0.00 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --sleep_level 1 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer true \
    --logging_steps 1 \
    --recompute_granularity none \
    --gradient_accumulation_fusion false \
    --finetune \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim \
    --no_save_rng \
    --save_steps 20 \
    --attention_backend flash \
    --moe_expert_capacity_factor 2 \
    --temperature 1.0 \
    --padding_free false \
    --sequence_parallel true \
    --log_completions true \
    --report_to tensorboard
```

## Known issues

- **Reinforcement learning**: If vLLM is the inference engine, use vLLM ≥ 0.11.0. It is recommended to use ROCm 7.0 or the image we provide to avoid the sleep mode memory leak issue.
- **MoE training**: Set `NVTE_USE_CUTLASS_GROUPED_GEMM=1` to reduce occasional GPU hangs.
