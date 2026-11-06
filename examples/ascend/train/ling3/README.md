# Ling-3.0-tiny 在 Ascend NPU 上进行 FSDP2 LoRA SFT

本示例基于 ms-swift 的 Transformers 训练流程，对 [inclusionAI/Ling-3.0-tiny](https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny) 进行 LoRA 监督微调，使用 FSDP2 分布式训练。模型类型为 `bailing_hybrid`，对话模板为 `ling3`。

本目录包含：

- `ling3_tiny_lora_fsdp2.sh`：单机 8 卡 FSDP2 LoRA 训练脚本。
- `fsdp2.json`：FSDP2 分布式训练配置。

## 1. 环境准备

本示例在 Ascend A3 服务器上完成验证，软件版本如下：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.10.0 |
| torch_npu | 2.10.0.post4 |
| Transformers | 4.57.1 / 5.14.1 |
| CANN | 9.1.0 |
| triton-ascend | 3.2.2 |

使用前需配置与设备匹配的 CANN、PyTorch、torch_npu 和 triton-ascend 环境。上述版本为已验证组合，其他版本组合需另行验证。

### 安装 Flash Linear Attention

Ling-3.0-tiny 的 KDA 计算依赖 [Flash Linear Attention（FLA）](https://github.com/fla-org/flash-linear-attention)。本示例使用以下固定源码版本：

```bash
git clone https://github.com/fla-org/flash-linear-attention.git
cd flash-linear-attention
git checkout 35dceaee5408e69a555fec34cb215c93c375dabe
pip install -e . --no-deps
```

`--no-deps` 用于保留现有的 NPU 软件环境，安装前需确保其他 Python 依赖已配置完成。

该 Commit 为已验证的上游开发版本，并非正式 Release。其他 FLA 版本尚未完成本示例的兼容性验证。

## 2. 模型与数据准备

### 模型

脚本默认使用模型 ID `inclusionAI/Ling-3.0-tiny`。ms-swift 会从 ModelScope 自动下载模型并使用本地缓存，无需提前准备模型目录。

如需从 Hugging Face 下载，可在启动命令后添加 `--use_hf true`，并确保训练节点能够访问 Hugging Face。

离线运行时，也可以通过 `MODEL` 指定已下载模型的本地目录。该目录需要包含完整的模型权重、配置文件、Tokenizer 及官方模型代码。

### 数据

脚本默认使用以下公开数据集：

- `AI-ModelScope/alpaca-gpt4-data-zh`
- `AI-ModelScope/alpaca-gpt4-data-en`
- `swift/self-cognition`

也可以通过 `--dataset` 指定自己的数据集。数据格式参考 ms-swift 的 [自定义数据集文档](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html)。

## 3. 启动训练

脚本默认使用单机 8 卡 FSDP2，在 ms-swift 仓库根目录执行：

```bash
bash examples/ascend/train/ling3/ling3_tiny_lora_fsdp2.sh
```

默认使用设备 `0–7`。如需选择其他设备，可通过 `ASCEND_RT_VISIBLE_DEVICES` 指定，设备数量需与 `fsdp2.json` 中的 `num_processes=8` 保持一致。

使用本地模型时：

```bash
MODEL=/path/to/Ling-3.0-tiny \
bash examples/ascend/train/ling3/ling3_tiny_lora_fsdp2.sh
```

脚本默认从 `/usr/local/Ascend/ascend-toolkit` 加载 CANN 环境。如果安装路径不同，可通过 `CANN_PATH` 指定。

### 默认训练参数

| 参数 | 默认值 |
| --- | --- |
| 训练方式 | LoRA，rank 8，alpha 32 |
| 分布式策略 | FSDP2，单机 8 卡 |
| 计算精度 | BF16 |
| Attention Backend | FlashAttention 2 |
| 每卡 Batch Size | 1 |
| 梯度累积步数 | 2 |
| 最大序列长度 | 2048 |
| Packing | true，Sequential |
| Gradient Checkpointing | false |
| 训练轮数 | 2 |

训练结果默认保存在 `output/Ling3.0-Tiny-lora-fsdp2`。

脚本支持在启动命令后追加 ms-swift 参数。首次运行可以先执行少量训练步，确认训练流程正常：

```bash
bash examples/ascend/train/ling3/ling3_tiny_lora_fsdp2.sh \
    --max_steps 3 \
    --output_dir output/ling3-smoke
```

## 4. Padding 与 Packing

当前支持普通左右 Padding 和基于 FlashAttention 2 Varlen 的 Padding-free Packing。

### 普通 Padding

关闭 Packing 后，可使用普通右 Padding：

```bash
bash examples/ascend/train/ling3/ling3_tiny_lora_fsdp2.sh \
    --packing false \
    --padding_free false \
    --padding_side right
```

将 `right` 改为 `left` 即可使用左 Padding。

### Padding-free Packing

脚本默认启用 Sequential Padding-free Packing，配置如下：

```ini
packing=true
packing_length=2048
packing_strategy=sequential
attn_impl=flash_attn
```

当前支持的 Packing 路径使用 FlashAttention 2 Varlen，并要求明确的序列边界元数据。

以下组合暂不支持：

- Packing 与 KV Cache 或 MTP 同时使用；
- Packing 与 Context Parallel 同时使用；
- 使用 SDPA 或 Eager Attention 进行 Packing；
- 使用自定义 4D Attention Mask 进行 Packing。

此外，普通左右 Padding 支持无历史 KV Cache 的训练路径。训练模型入口不接受外部自定义 4D Attention Mask；带历史状态的 Padded KDA Decode 不在当前支持范围内。

## 5. 测试与注意事项

### 测试

可在 ms-swift 仓库根目录执行 Ling3 相关的 CPU 回归测试：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
    tests/models/test_ling3_rope.py \
    tests/models/test_ling3_compat.py \
    tests/models/test_ling3_packing.py \
    tests/models/test_ling3_loader.py \
    -v
```

部分集成测试需要官方模型源码，可以通过 `LING3_MODEL_SOURCE` 指定本地模型目录。未提供时，相应测试会跳过。

CPU 测试不执行真实 Ascend Kernel，完整训练还需在实际 NPU 环境中验证。

### 注意事项

- **模型范围**：当前训练适配面向官方 Ling-3.0-tiny，不包含其他 Bailing V3 模型。
- **Transformers**：支持已验证的 4.57.1 和 5.14.1，使用默认 RoPE 配置。
- **Gradient Checkpointing**：脚本默认关闭。如果设备内存不足，可通过 `--gradient_checkpointing true` 开启。
- **Checkpoint**：脚本默认 `save_only_model=true`，不保存完整的优化器和训练调度状态。如需完整恢复训练，应在训练开始前设置 `--save_only_model false`，并使用兼容的训练配置恢复。
- **MTP**：当前示例不支持 MTP 训练。

本示例提供的是单机 8 卡 BF16 FSDP2 LoRA SFT 配置，不包含多机训练或全参数微调。
