# DeepSeek-V4.1 LoRA example

[中文](README_CN.md)

## Software versions and sources

The following stack was used for the single-host, four-layer, 2K-packed, 10-step validation. Pin source components to the listed branches and commits; package version numbers alone do not identify the tested implementation.

| Component | Validated version / source |
| --- | --- |
| Python | 3.12.15 |
| CANN | 9.2.0 |
| PyTorch / torch-npu | 2.10.0 / 2.10.0.post2 |
| Transformers / PEFT | 5.12.1 / 0.20.0 |
| datasets / accelerate | 4.8.4 / 1.15.0 |
| safetensors / ModelScope | 0.8.0 / 1.40.2 |
| NumPy | 1.26.4 |
| Megatron-Core | [tastelikefeet/Megatron-LM](https://github.com/tastelikefeet/Megatron-LM/tree/827e940e602b8133eb8d5525d5ebfd59e6923ec5), `dsv41-pr7224-engram-local`, `827e940e602b8133eb8d5525d5ebfd59e6923ec5` (0.19.0+827e940) |
| MegatronAdaptor | [Ascend/MegatronAdaptor](https://gitcode.com/Ascend/MegatronAdaptor), `core_r0.19.0`, `c7e52b2c8482105e6ec7fcd8cdef364e7c7b22c5` |
| TransformerEngineNPU | [Ascend/TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU), `main`, `d5c6a4d38db1ebab909518f92457559ddd604c4e` |
| ms-swift | [PR #10501](https://github.com/modelscope/ms-swift/pull/10501), `addsubmuldiv/ms-swift:dsv41`, `38aafdf94a73e6f453d5a8ea30f09be98df92659` (4.7.0.dev0) |
| mcore-bridge | [PR #251](https://github.com/modelscope/mcore-bridge/pull/251), `addsubmuldiv/mcore-bridge:dsv41`, `7e6c6cf4485c1e72ab1454152976bb8538c8ea24` (1.7.0.dev0) |

The Swift and bridge commits identify the source snapshots for this adaptation. Use the implementation in this PR and the companion bridge PR rather than PyPI packages with matching version numbers. MCore must come from the listed fork containing the V4.1 / native Engram implementation; upstream `megatron-core==0.19.0` is not a substitute. Pin MegatronAdaptor and TransformerEngineNPU to the listed commits rather than following their latest `main`. Follow each project's source installation instructions, use editable installs, and verify that runtime imports resolve to those checkouts.

The historical two-host full-model run used the same MCore, MegatronAdaptor, TransformerEngineNPU and base dependencies, but locally modified snapshots of Swift `6ddd3c4973fa4c2af4da22c9047c9859fb4640e0` ([PR #10087](https://github.com/modelscope/ms-swift/pull/10087)) and bridge `f79dff30145d10e070f5514770b60fe66dd6fee8` ([PR #189](https://github.com/modelscope/mcore-bridge/pull/189)), including the earlier offload argument and NPU loading adaptations. Checking out those two commits alone does not reproduce that run. The current two-host configuration using the shared environment variable has not been rerun.

## Single-host reduced-model validation

The [example script](megatron_lora.sh) uses TP=1, PP=1, CP=1 and EP=8 to train attention-projection LoRA parameters on NPU. `NGRAM_EMBEDDING_CPU_OFFLOAD=1` keeps frozen, EP-sharded Engram tables in CPU memory; other computation stays on NPU. The defaults enable 2K packing and save Megatron-format adapter checkpoints without merging the full model.

Load the CANN environment first and prepare the NPU stack listed above. Standard GPU installation commands do not replace these NPU components.

```bash
MODEL=/path/to/deepseek-v41-checkpoint \
ENGRAM_TOKENIZER_MAP=/path/to/engram_tokenizer_map.json \
DATASET=/path/to/train.jsonl \
OUTPUT_DIR=megatron_output/deepseek-v41-npu-lora \
bash examples/ascend/train/deepseek_v41/megatron_lora.sh
```

`MODEL` is a required local checkpoint directory. If `ENGRAM_TOKENIZER_MAP` is unset, the script reads `engram_tokenizer_map.json` from that directory. The map must match the checkpoint's tokenizer, original layer IDs and vocabulary; generate it with the matching MCore `tools/engram/generate_tokenizer_map.py`. `DATASET` accepts local data or defaults to the example dataset in the script. Adjust `TRAIN_ITERS`, `SAVE_STEPS`, `MAX_LENGTH` and `GLOBAL_BATCH_SIZE`. Configure visible devices and communication interfaces for your host; no interface or NUMA layout is hard-coded.

The single-host script completed 10 training steps in this configuration and saved adapter checkpoints at steps 5 and 10.

Validation covers one host with 8 Ascend 950DT devices and a four-layer text checkpoint retaining the original hidden size, expert count and one Engram table. It includes ordinary attention, MoE, hyper-connections and a CSA2 ratio-2 source/reuse pair. It excludes ratio-1/candidate paths, DSpark, vision, the complete 40-layer model and multi-node training. CPU memory must fit all local Engram shards plus model-loading temporaries. Reduced-model results do not establish full-model compatibility; see below for historical two-host full-model validation. This script does not validate full-parameter training or full HF model export.

## Two-host full-depth model

The [two-host script](megatron_lora_multinode.sh) follows the previously successful configuration for the complete 40-layer text model: 2 hosts with 8 NPUs each, TP=1, PP=4, CP=1, EP=4, micro batch size 1 and global batch size 4. It defaults to sequence length 256, disables packing and padding-free execution, and uses `unfused` attention. `NGRAM_EMBEDDING_CPU_OFFLOAD=1` freezes Engram tables and keeps them on CPU.

Load the matching CANN/NPU environment on both hosts and prepare identical checkpoints, tokenizer maps and datasets. Run the following from the repository root on each host:

```bash
# Node 0
NODE_RANK=0 MASTER_ADDR=<node-0-communication-IP> \
MODEL=/path/to/full-deepseek-v41-checkpoint \
ENGRAM_TOKENIZER_MAP=/path/to/engram_tokenizer_map.json \
DATASET=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/shared-output \
bash examples/ascend/train/deepseek_v41/megatron_lora_multinode.sh

# Node 1: use the same MASTER_ADDR, MASTER_PORT and OUTPUT_DIR as node 0
NODE_RANK=1 MASTER_ADDR=<node-0-communication-IP> \
MODEL=/path/to/full-deepseek-v41-checkpoint \
ENGRAM_TOKENIZER_MAP=/path/to/engram_tokenizer_map.json \
DATASET=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/shared-output \
bash examples/ascend/train/deepseek_v41/megatron_lora_multinode.sh
```

Replace `<node-0-communication-IP>` with an address reachable from both hosts. Set `HCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` for each host. Use a shared `OUTPUT_DIR` accessible from both hosts; automatic version suffixes are disabled so both nodes save the same training run. The successful run also used a NUMA Python launcher with CPU/memory binding matched to each host's NPU topology. Set `PYTHON_EXEC` to such a launcher when needed, using the actual host topology rather than another machine's NUMA IDs. Host memory must fit local Engram shards and loading temporaries.

The historical run completed 10 steps on 16 Ascend 950DT devices across two hosts, using the full 40-layer text checkpoint and short samples with a maximum length of 256. Checkpoints at steps 5 and 10 each contained 16 shards. All 120 LoRA B tensors were finite and nonzero, and changed between the two checkpoints. This validation used the earlier offload argument; the script now uses the shared environment variable and has not been rerun on two hosts with the current code. Long sequences, vision, full-parameter training and full HF export were outside that validation.
