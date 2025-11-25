# Circle-RoPE for Qwen2.5-VL - V2 版本

## 版本说明

### V1 vs V2

- **V1** (`modular_qwen2_5_vl_circle_rope.py`): 基于旧版 transformers 架构
- **V2** (`modular_qwen2_5_vl_circle_rope_v2.py`): 兼容最新 transformers 架构 ✅

### V2 主要改进

1. **架构适配**：
   - ✅ `get_rope_index` 方法移至 `Qwen2_5_VLModel` 层
   - ✅ `rope_deltas` 缓存在 Model 层（非 ForConditionalGeneration）
   - ✅ 支持 torch.dynamo 编译模式
   - ✅ 修复了 tensor.item() 的 bug

2. **新版架构结构**：
   ```
   Qwen2_5_VLForConditionalGeneration_CircleRoPE_V2
   ├── self.model (Qwen2_5_VLModel_CircleRoPE_V2)
   │   ├── self.visual         # 视觉编码器
   │   └── self.language_model # 语言模型
   └── self.lm_head
   ```

## 使用方法

### 1. 基本训练配置

创建训练配置文件 `train_config_v2.json`:

```json
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "model_type": "qwen2_5_vl_circle_rope_v2",
  "custom_register_path": ["circle_rope/register_v2.py"],
  "model_config_override": {
    "architectures": ["Qwen2_5_VLForConditionalGeneration_CircleRoPE_V2"],
    "circle_rope": {
      "circle_r": 10000,
      "base": 10000,
      "mrope_section": [16, 24, 24]
    }
  },
  "dataset": "your_dataset",
  "num_train_epochs": 1,
  "per_device_train_batch_size": 1,
  "learning_rate": 1e-5,
  "output_dir": "output/qwen2_5_vl_circle_rope_v2"
}
```

### 2. AGE 模式配置

AGE (Alternating Grouped Encoding) 模式可以在不同层使用不同的 RoPE 索引策略：

```json
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "model_type": "qwen2_5_vl_circle_rope_v2",
  "custom_register_path": ["circle_rope/register_v2.py"],
  "model_config_override": {
    "architectures": ["Qwen2_5_VLForConditionalGeneration_CircleRoPE_V2"],
    "circle_rope": {
      "circle_r": 10000,
      "base": 10000,
      "mrope_section": [16, 24, 24],
      "AGE_mode": "strategy_2"
    }
  }
}
```

**AGE 策略说明**（假设 36 层）：
- `strategy_2`: 前 18 层用 Circle-RoPE，后 18 层用原始索引
- `strategy_3`: 前 18 层用原始索引，后 18 层用 Circle-RoPE
- `strategy_4`: 奇数层用 Circle-RoPE，偶数层用原始索引

### 3. 命令行训练

```bash
# 使用配置文件
swift sft --load_args train_config_v2.json

# 或直接命令行
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type qwen2_5_vl_circle_rope_v2 \
    --custom_register_path circle_rope/register_v2.py \
    --model_config_override '{"architectures":["Qwen2_5_VLForConditionalGeneration_CircleRoPE_V2"],"circle_rope":{"circle_r":10000,"base":10000,"mrope_section":[16,24,24]}}' \
    --dataset your_dataset \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --output_dir output/qwen2_5_vl_circle_rope_v2
```

## Circle-RoPE 参数说明

### `circle_rope` 配置

```json
{
  "circle_rope": {
    "circle_r": 10000,        // Circle-RoPE 的半径参数
    "base": 10000,            // RoPE 的基础频率
    "mrope_section": [16, 24, 24],  // 多分辨率 RoPE 分段
    "AGE_mode": "strategy_2"  // (可选) AGE 策略
  }
}
```

### mrope_section 说明

Qwen2.5-VL 使用多分辨率 RoPE (M-RoPE)，将 head_dim 分为 3 段：
- 第1段 (16维): 时间维度
- 第2段 (24维): 高度维度
- 第3段 (24维): 宽度维度

总计: 16 + 24 + 24 = 64 (head_dim)

## 文件结构

```
circle_rope/
├── modular_qwen2_5_vl_circle_rope.py      # V1 (旧版)
├── modular_qwen2_5_vl_circle_rope_v2.py   # V2 (新版) ✅
├── circle_rope_imp.py                      # Circle-RoPE 核心实现
├── register.py                             # V1 注册文件
├── register_v2.py                          # V2 注册文件 ✅
└── README_v2.md                            # 本文档
```

## 核心类说明

### 1. `Qwen2_5_VLConfig_CircleRoPE_V2`
扩展配置类，添加 `circle_rope` 参数。

### 2. `Qwen2_5_VLModel_CircleRoPE_V2`
核心模型类，包含：
- `get_rope_index()`: 计算 Circle-RoPE 索引
- `_get_circle_index()`: Circle-RoPE 索引计算
- `_get_m_index()`: 原始 m-index 计算（用于 AGE 模式）
- `rope_deltas`: 位置索引缓存

### 3. `Qwen2_5_VLModel_AGE_V2`
AGE 模式支持类，允许不同层使用不同的位置编码策略。

### 4. `Qwen2_5_VLForConditionalGeneration_CircleRoPE_V2`
主模型类，根据配置自动选择标准或 AGE 模式。

## 与旧版的区别

| 特性 | V1 (旧版) | V2 (新版) |
|------|-----------|-----------|
| get_rope_index 位置 | ForConditionalGeneration | Qwen2_5_VLModel ✅ |
| rope_deltas 缓存 | ForConditionalGeneration | Qwen2_5_VLModel ✅ |
| 架构层级 | 2层 (model + visual) | 3层 (model.visual + model.language_model) ✅ |
| tensor.item() bug | 存在 | 已修复 ✅ |
| torch.dynamo 支持 | ❌ | ✅ |

## 测试与验证

### 快速验证

```python
from transformers import AutoConfig
from circle_rope.modular_qwen2_5_vl_circle_rope_v2 import (
    Qwen2_5_VLConfig_CircleRoPE_V2,
    Qwen2_5_VLForConditionalGeneration_CircleRoPE_V2
)

# 创建配置
config = Qwen2_5_VLConfig_CircleRoPE_V2.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    trust_remote_code=True
)
config.circle_rope = {
    "circle_r": 10000,
    "base": 10000,
    "mrope_section": [16, 24, 24]
}

# 初始化模型
model = Qwen2_5_VLForConditionalGeneration_CircleRoPE_V2(config)
print(f"✅ 模型初始化成功: {type(model.model).__name__}")
```

## 常见问题

### Q1: 如何确定使用 V1 还是 V2？

**A:** 检查你的 transformers 版本：
```bash
python -c "import transformers; print(transformers.__version__)"
```
- transformers >= 4.49: 使用 **V2** ✅
- transformers < 4.49: 使用 V1

### Q2: 如何从 V1 迁移到 V2？

**A:** 只需修改配置文件：
1. `model_type`: `qwen2_5_vl_circle_rope` → `qwen2_5_vl_circle_rope_v2`
2. `custom_register_path`: `register.py` → `register_v2.py`
3. `architectures`: 添加 `_V2` 后缀

### Q3: V2 支持哪些 Qwen2.5-VL 模型？

**A:** 支持所有 Qwen2.5-VL 系列模型：
- Qwen2.5-VL-2B-Instruct
- Qwen2.5-VL-7B-Instruct
- Qwen2.5-VL-72B-Instruct

## 贡献者

- 原始 Circle-RoPE 实现：基于旧版 transformers
- V2 升级：适配最新 transformers 架构

## License

与 ms-swift 保持一致。
