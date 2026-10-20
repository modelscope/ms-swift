# Olmo-3-7B-Think 在 Ascend NPU 上进行 LoRA SFT

本示例基于 ms-swift 的原生 Transformers 训练流程，使用 `swift sft` 对 `allenai/Olmo-3-7B-Think` 进行 LoRA 监督微调。模型类型为 `olmo3`，对话模板为 `olmo3_think`。

本目录包含：

- `sft_lora.sh`：单机 16 卡 LoRA 训练示例。
- `README.md`：环境准备、数据格式、运行方式及适配限制。

## 1. 环境

本示例提供的运行环境版本如下：

| 组件 | 版本 |
| --- | --- |
| PyTorch | `2.10.0+cpu` |
| torch_npu | `2.10.0.post4` |
| Transformers | `4.57.1` |
| CANN | `9.1.0` |

运行平台为 Linux Ascend NPU 服务器。设备型号、单卡内存、驱动、Python 版本以及训练性能请以实际部署环境为准。上述版本记录不代表其他版本组合也已验证。

先按服务器上的实际安装路径加载 CANN 环境并激活 Python 环境，然后在**包含 Olmo3 适配提交的 ms-swift 仓库根目录**安装源码：

```bash
# torch、torch_npu 和 CANN 应已按上述环境配置完成。
python -m pip install -e .
python -m pip check
```

安装时应保留上表中的运行时版本；如依赖解析改变了版本，需先处理依赖约束再运行。Transformers 需包含 Olmo3 实现，本示例使用 `4.57.1`。

检查 Python 包版本及 NPU 可见性：

```bash
python - <<'PY'
import torch
import torch_npu
import transformers

print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('transformers:', transformers.__version__)
print('NPU available:', torch.npu.is_available())
print('NPU count:', torch.npu.device_count())
PY

npu-smi info
```

`torch` 版本中的 `+cpu` 后缀本身不能用于判断 NPU 是否可用；本示例通过 `torch_npu` 使用 NPU，应以导入结果、`torch.npu.is_available()` 和设备检测结果为准。

## 2. 模型与数据准备

### 模型

提前准备 `allenai/Olmo-3-7B-Think` 的完整 Hugging Face 格式模型目录，包括配置、tokenizer、对话模板及全部权重文件，例如 `config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`*.safetensors` 和分片索引（若有）。仅有配置文件无法进行完整模型训练。

将 `sft_lora.sh` 中的 `MODEL_PATH` 替换为该目录的绝对路径。本地完整模型目录适合离线运行，也能避免不同模型下载源带来的差异。

### 数据

准备 UTF-8 编码的 JSONL 文件，每行一条样本，使用 ms-swift 支持的 `messages` 文本对话格式：

```jsonl
{"messages":[{"role":"user","content":"计算 12 × 13。"},{"role":"assistant","content":"<think>12 × 13 = 12 × (10 + 3) = 120 + 36 = 156。</think>\n答案是 156。"}]}
{"messages":[{"role":"user","content":"一盒有 6 支笔，4 盒共有多少支？"},{"role":"assistant","content":"<think>每盒 6 支，4 盒共有 6 × 4 = 24 支。</think>\n共有 24 支笔。"}]}
```

以上两条仅用于展示格式。用于 Think 风格训练时，assistant 内容应包含完整的 `<think>...</think>` 思考过程和最终答案。无需手动添加 `<|im_start|>`、`<|im_end|>`、EOS 或重复的思考前缀，模板会处理对话边界。

可提供文本 `system` 消息或通过 `--system` 设置系统提示；默认系统提示和无工具声明由 `olmo3_think` 模板提供。支持多轮纯文本对话，当前不支持工具调用和多模态输入。原生 OLMo 的 `functions`、`function_calls`、`environment` 协议未适配，应在使用前转换为受支持的文本对话格式。

将 `DATASET_PATH` 替换为训练 JSONL 文件的绝对路径。当前目录不附带样本数据文件。

## 3. 启动训练

在运行前编辑 `sft_lora.sh` 中的三个路径：

```bash
MODEL_PATH="/path/to/Olmo-3-7B-Think"
DATASET_PATH="/path/to/train.jsonl"
OUTPUT_DIR="/path/to/output/olmo3-lora"
```

脚本默认使用单机 16 卡：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NPROC_PER_NODE=16
```

请按实际可用设备修改**脚本内**的设备列表和进程数，保证二者匹配。例如单卡设为 `0` / `1`，八卡设为 `0,1,2,3,4,5,6,7` / `8`。脚本会直接赋值这些变量及三个路径变量，运行前在终端设置同名变量不能覆盖它们；脚本也未转发额外命令行参数，修改训练参数需直接编辑脚本。

在仓库根目录执行：

```bash
bash examples/ascend/train/olmo3/sft_lora.sh
```

脚本使用 `MASTER_ADDR=127.0.0.1`、`MASTER_PORT=29500` 和 `HCCL_CONNECT_TIMEOUT=600`，适用于单机运行；端口被占用时需修改脚本中的端口。示例未配置多机启动。

### 默认训练参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `model_type` / `template` | `olmo3` / `olmo3_think` | 模型及 Think 对话模板 |
| `tuner_type` | `lora` | LoRA 微调 |
| `lora_rank` / `lora_alpha` | `8` / `32` | LoRA 配置 |
| `target_modules` | `all-linear` | 由 ms-swift 解析实际目标线性层 |
| `torch_dtype` | `bfloat16` | BF16 训练 |
| `attn_impl` | `sdpa` | 保留滑动窗口与全注意力语义 |
| `per_device_train_batch_size` | `1` | 每卡微批大小 |
| `gradient_accumulation_steps` | `2` | 梯度累积步数 |
| `gradient_checkpointing` | `true` | 启用梯度检查点以降低内存占用 |
| `learning_rate` | `1e-4` | 初始学习率 |
| `lr_scheduler_type` / `warmup_ratio` | `cosine` / `0.05` | 学习率调度及预热比例 |
| `num_train_epochs` | `1` | 训练轮数 |
| `max_length` | `4096` | 最大样本长度，单位为 token |
| `padding_free` | `false` | 使用带 padding 的训练路径 |
| `dataloader_num_workers` / `dataset_num_proc` | `2` / `4` | 数据加载及预处理进程数 |
| `logging_steps` | `5` | 日志间隔 |
| `save_steps` / `save_total_limit` | `200` / `2` | checkpoint 保存间隔及保留数量 |
| `report_to` | `none` | 不接入外部实验记录平台 |

单机有效全局 batch size 为 `进程数 × 每卡 batch size × 梯度累积步数`，默认是 `16 × 1 × 2 = 32`。调整卡数时可结合训练需求调整梯度累积步数。

当前脚本使用多进程数据并行，**没有启用 DeepSpeed/ZeRO，也不会按卡数自动启用**。数据并行会在各设备保留模型副本，增加卡数不能直接解决单卡模型内存不足的问题。实际内存占用与设备、序列长度及运行时有关。

## 4. 训练前验证

可先在仓库根目录执行适配相关测试（需安装 `pytest`）：

```bash
python -m pytest tests/general/test_olmo3_template.py tests/models/test_olmo3.py -v
```

仅执行真实 NPU 硬件用例：

```bash
python -m pytest tests/models/test_olmo3.py -v -k 'npu_bf16 or npu_sdpa'
```

硬件用例使用小模型配置，覆盖滑动窗口与 padding 下的 eager/SDPA 对比，以及 BF16、梯度检查点和 LoRA 两步训练；无需下载完整 7B 权重。没有可用 NPU 时会跳过，`skipped` 不代表验证通过。

首次运行完整模型时，可先复制一份脚本用于短训练验证：按实际设备修改卡数，将 `--max_length` 改为 `256`，将 `--gradient_accumulation_steps` 改为 `1`、`--logging_steps` 改为 `1`，并在 `swift sft` 参数中增加：

```bash
    --max_steps 2 \
    --save_strategy no \
    --eval_strategy no \
```

使用少量格式正确的数据，确认两个优化步完成、loss 为有限值且进程正常退出，再运行正式训练。短训练仅验证训练链路，不能证明模型质量、长序列精度或吞吐性能。

## 5. 适配说明与限制

- NPU 注意力支持 `sdpa` 和 `eager`，脚本使用 `sdpa`。Olmo3 混合使用滑动窗口注意力与全注意力，需保留逐层对应的 mask；当前适配拒绝 NPU FlashAttention 路径。不要通过修改 `layer_types` 或 `sliding_window` 绕过限制。
- NPU 算子适配覆盖满足条件时的 RMSNorm 和 SwiGLU MLP；RoPE 保留原生 FP32 计算路径。算子替换包含设备、数据类型或激活函数检查，不代表所有输入均使用融合算子。
- 当前不支持 `packing`、`padding_free` 和 `sequence_parallel_size > 1`；保持 `false`、`false`、`1`。多卡数据并行不受此限制。
- 本目录仅提供 LoRA SFT 示例，没有全参数训练脚本；本次模型适配不包含 Megatron 后端。

如出现 NPU 不可见，先检查 CANN 环境是否加载、`torch_npu` 是否能导入以及设备检测结果；如出现通信连接或端口错误，检查设备列表、进程数、端口及 HCCL 日志；如内存不足，可先减小序列长度并保留梯度检查点，再根据实际环境调整训练配置。
