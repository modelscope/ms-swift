# DeepSeek-V4.1 LoRA 示例

[English](README.md)

## 软件版本与源码

以下是单机 4 层、2K packing、10 步验证所用的软件栈。源码组件使用下表中的仓库和分支。

| 组件 | 验证版本 / 源码 |
| --- | --- |
| Python | 3.12.15 |
| CANN | 9.2.0 |
| PyTorch / torch-npu | 2.10.0 / 2.10.0.post2 |
| Transformers / PEFT | 5.12.1 / 0.20.0 |
| datasets / accelerate | 4.8.4 / 1.15.0 |
| safetensors / ModelScope | 0.8.0 / 1.40.2 |
| NumPy | 1.26.4 |
| Megatron-Core | [tastelikefeet/Megatron-LM](https://github.com/tastelikefeet/Megatron-LM/tree/dsv41-pr7224-engram-local), `dsv41-pr7224-engram-local` (0.19.0) |
| MegatronAdaptor | [Ascend/MegatronAdaptor](https://gitcode.com/Ascend/MegatronAdaptor), `core_r0.19.0` |
| TransformerEngineNPU | [Ascend/TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU), `main` |

MCore 使用上述包含 V4.1 / 原生 Engram 实现的 fork 分支。MegatronAdaptor 和 TransformerEngineNPU 分别使用表中的分支，按各项目的安装说明使用源码 editable 安装，并确认运行时导入的是这些 checkout。

## 单机减层验证

[示例脚本](megatron_lora.sh)使用 TP=1、PP=1、CP=1、EP=8，在 NPU 上训练注意力投影的 LoRA 参数。`NGRAM_EMBEDDING_CPU_OFFLOAD=1` 将 Engram 词表冻结并按 EP 分片保留在 CPU，其他计算仍在 NPU 上。默认开启 2K packing，保存 Megatron 格式的 adapter checkpoint（不合并完整模型）。

先加载 CANN 环境，并按上面的版本表准备 NPU 软件栈；普通 GPU 环境的安装命令不能直接替代这些 NPU 组件。

```bash
MODEL=/path/to/deepseek-v41-checkpoint \
ENGRAM_TOKENIZER_MAP=/path/to/engram_tokenizer_map.json \
DATASET=/path/to/train.jsonl \
OUTPUT_DIR=megatron_output/deepseek-v41-npu-lora \
bash examples/ascend/train/deepseek_v41/megatron_lora.sh
```

`MODEL` 为必填的本地 checkpoint 目录；未指定 `ENGRAM_TOKENIZER_MAP` 时，读取该目录下的 `engram_tokenizer_map.json`。此文件必须与 checkpoint 的 tokenizer、原始层号和词表匹配，可使用配套 MCore 的 `tools/engram/generate_tokenizer_map.py` 生成。`DATASET` 支持本地数据，也可省略以使用脚本中的示例数据集。可通过 `TRAIN_ITERS`、`SAVE_STEPS`、`MAX_LENGTH`、`GLOBAL_BATCH_SIZE` 调整训练规模。设备可见性及通信网卡应按机器环境设置，脚本不写死网卡或 NUMA 布局。

单机脚本在上述配置下完成了 10 个训练 step，并成功保存 step 5 和 step 10 的 adapter checkpoint。

验证范围为单机 8 张 Ascend 950DT、4 层文本 checkpoint，保留原始隐藏维度、专家规模和一个 Engram 词表，覆盖普通注意力、MoE、hyper-connections 和 CSA2 ratio-2 的 source/reuse 配对。它不覆盖 ratio-1/candidate、DSpark、视觉分支、完整 40 层模型或多机训练。CPU 需要容纳本机所有 Engram 词表分片以及模型加载的临时内存；不能把减层结果直接外推到完整模型；双机完整模型的历史验证见下文。此脚本也不验证全参训练或完整 HF 模型导出。

## 双机完整模型

[双机脚本](megatron_lora_multinode.sh)基于此前完整 40 层文本模型跑通的配置：2 台机器，每台 8 张 NPU，TP=1、PP=4、CP=1、EP=4，micro batch size 为 1，global batch size 为 4。默认序列长度 256，关闭 packing 和 padding-free，使用 `unfused` attention；Engram 词表通过 `NGRAM_EMBEDDING_CPU_OFFLOAD=1` 冻结并保留在 CPU。

两台机器先加载匹配的 CANN/NPU 环境，准备相同 checkpoint、tokenizer map 和数据集。分别在仓库根目录运行：

```bash
# 节点 0
NODE_RANK=0 MASTER_ADDR=<节点0的通信IP> \
MODEL=/path/to/full-deepseek-v41-checkpoint \
ENGRAM_TOKENIZER_MAP=/path/to/engram_tokenizer_map.json \
DATASET=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/shared-output \
bash examples/ascend/train/deepseek_v41/megatron_lora_multinode.sh

# 节点 1：MASTER_ADDR、MASTER_PORT、OUTPUT_DIR 与节点 0 相同
NODE_RANK=1 MASTER_ADDR=<节点0的通信IP> \
MODEL=/path/to/full-deepseek-v41-checkpoint \
ENGRAM_TOKENIZER_MAP=/path/to/engram_tokenizer_map.json \
DATASET=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/shared-output \
bash examples/ascend/train/deepseek_v41/megatron_lora_multinode.sh
```

将 `<节点0的通信IP>` 替换为两台机器均可访问的实际地址，并按各自机器设置 `HCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME`。`OUTPUT_DIR` 使用两台机器均可访问的共享目录；脚本关闭自动版本后缀，确保两个节点写入同一次训练的 checkpoint。此前成功运行还使用了按本机 NPU/CPU 拓扑绑定的 NUMA Python launcher，可通过 `PYTHON_EXEC` 指定这类 launcher；需要按实际拓扑配置，不能直接照搬其他机器的 NUMA 编号。CPU 内存需要容纳本机 Engram 分片和加载临时数据。

此前在双机 16 张 Ascend 950DT、完整 40 层文本 checkpoint、256 长度短样本数据下完成 10 步，step 5 和 10 各保存了 16 个 checkpoint 分片，120 个 LoRA B 张量均为有限非零值，并在两个 checkpoint 之间发生更新。这是旧 offload 参数入口下的历史验证；本脚本已换用统一环境变量，尚未在当前代码上重跑双机。长序列、视觉分支、全参训练和完整 HF 导出不在该次验证范围内。
