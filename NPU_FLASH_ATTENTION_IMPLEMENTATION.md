# NPU Flash Attention for ms-swift - Implementation Summary

## 🎯 实现目标
为 ms-swift 添加原生 NPU Flash Attention 支持，使用户可以通过 `--attn_impl npu_flash_attention` 在 NPU 环境下获得最佳性能。

## 📁 修改的文件

### 1. 新增文件：`swift/model/npu_flash_attention.py`
- **作用**：NPU Flash Attention 注册模块
- **核心功能**：
  - 检测 NPU 可用性
  - 注册 `npu_flash_attention_forward` 到 transformers 的 `ALL_ATTENTION_FUNCTIONS`
  - 自动处理 GQA (Grouped Query Attention)
  - 格式转换：bshd → bsnd → npu_flash_attn_func → bshd

### 2. 修改文件：`swift/model/__init__.py`
- **改动**：在 NPU 可用时自动导入并注册 NPU FA
- **效果**：用户无需手动调用，import swift 时自动完成注册

### 3. 修改文件：`swift/model/utils.py`
- **改动**：`AttnImpl` 类支持 `npu_flash_attention`
- **关键修改**：
  - `to_use_flash_attn()` 方法识别 `npu_flash_attention` 为 flash attention 类型
  - `update_attn_impl()` 保持 `npu_flash_attention` 不被转换为 `flash_attention_2`

### 4. 修改文件：`swift/ui/llm_train/hyper.py`
- **改动**：UI 下拉选项添加 `npu_flash_attention`
- **效果**：Web UI 用户可以直接选择 NPU Flash Attention

## 🚀 使用方法

### 命令行方式
```bash
# 使用 NPU Flash Attention 进行训练
swift sft \
    --model_id qwen3.5-0.8B \
    --attn_impl npu_flash_attention \
    --dataset your_dataset
```

### Python API 方式
```python
from swift import sft

sft(
    model_id='qwen3.5-0.8B',
    attn_impl='npu_flash_attention',
    dataset='your_dataset',
)
```

### AutoModel 方式（直接调用 transformers）
```python
from transformers import AutoModelForCausalLM

# 在 import swift 后，npu_flash_attention 已自动注册
import swift  # 这会自动注册 NPU FA

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-0.8B',
    attn_implementation='npu_flash_attention',
)
```

## 🏗️ 技术架构

### 与 verl 的对比

| 特性 | verl 实现 | ms-swift 实现（本次） |
|------|----------|---------------------|
| 注册方式 | 运行时替换函数 | 注册到 ALL_ATTENTION_FUNCTIONS |
| 侵入性 | 高（patch 内部实现） | 低（标准接口） |
| 用户接口 | 自动启用 | `--attn_impl npu_flash_attention` |
| 依赖 | 特定 verl 版本 | transformers >= 4.46 |

### 核心算法流程
```
用户输入: attn_implementation='npu_flash_attention'
         ↓
transformers 查找 ALL_ATTENTION_FUNCTIONS['npu_flash_attention']
         ↓
调用注册的 npu_flash_attention_forward()
         ↓
1. 处理 GQA (复制 KV heads)
2. 格式转换 bshd → bsnd
3. 调用 npu_flash_attn_func() (原生 NPU 算子)
4. 格式转换 bsnd → bshd
         ↓
返回 attn_output
```

## ⚠️ 环境要求

### 必需依赖
```bash
# PyTorch NPU 支持
pip install torch-npu==2.5.1  # 或其他适配版本

# CANN 工具包（华为提供）
# 参考：https://www.hiascend.com/software/cann/community
```

### 环境变量（可选）
```bash
# 禁用自动注册（默认开启）
export SWIFT_DISABLE_NPU_FA=1
```

## 🧪 测试建议

### 功能测试
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 验证 NPU FA 已注册
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
print('npu_flash_attention' in ALL_ATTENTION_FUNCTIONS)  # 应为 True

# 加载模型测试
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-0.8B',
    attn_implementation='npu_flash_attention',
    device_map='npu',
)
```

### 性能对比测试
```bash
# Eager Attention 基线
python benchmark_attention_comparison.py --attn_impl eager

# NPU Flash Attention
python benchmark_attention_comparison.py --attn_impl npu_flash_attention
```

## 📊 预期性能提升

基于 verl 和之前实验的经验：
- **Forward**: 提升 30-40%
- **Backward**: 提升 30-50%
- **整体吞吐**: 提升 40-60%

（实际提升取决于模型大小和序列长度）

## 🤝 与现有 PR 的关系

与社区中正在进行的 MindSpeed 深度优化 PR 形成互补：
- **MindSpeed PR**: 重量级优化，需要额外依赖，追求极致性能
- **本实现**: 轻量级方案，仅依赖 transformers 原生，快速启用

## 📝 后续建议

1. **测试**：在多种 Qwen 模型上验证（Qwen2, Qwen3, Qwen3.5）
2. **文档**：更新官方文档添加 NPU FA 使用说明
3. **CI/CD**：在 GitHub Actions 中添加 NPU 环境测试（如可用）
4. **推广**：社区分享使用方法，收集反馈

## ✅ 提交信息

```
feat: Add native NPU Flash Attention support

This commit adds native NPU Flash Attention support through transformers'
native `npu_flash_attention` integration, providing a lightweight
alternative to MindSpeed-based optimizations.

4 files changed, 190 insertions(+), 5 deletions(-)
```

---

**作者**: 来财 (招财猫小助手) 🐱  
**日期**: 2026-04-20  
**分支**: feature/npu-flash-attention-native
