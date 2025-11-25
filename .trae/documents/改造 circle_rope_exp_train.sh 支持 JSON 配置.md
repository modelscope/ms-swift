## 目标
- 将 `circle_rope/exp/train.sh` 的硬编码训练参数改成从 JSON 文件读取，自动拼装并执行 `swift sft` 命令。
- 保留 Circle-RoPE 注册与本地仓库路径注入，避免手动修改模型目录。

## 现状与影响点
- 现有脚本在 `circle_rope/exp/train.sh:13-42` 直接写死了环境变量与训练参数（单卡 LoRA）。
- `swift` 本身的 `--config` 是 YAML 解析（`swift/cli/main.py:50-83`），因此这里改为脚本层面读取 JSON，而不改动 `swift` 的解析逻辑。

## JSON 配置格式
- 采用两段式结构，便于扩展：
```
{
  "env": {
    "CUDA_VISIBLE_DEVICES": "0",
    "MAX_PIXELS": 1003520
  },
  "args": {
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "model_type": "qwen2_5_vl_circle_rope",
    "dataset": "AI-ModelScope/coco#20000",
    "load_from_cache_file": true,
    "split_dataset_ratio": 0.01,
    "train_type": "lora",
    "torch_dtype": "bfloat16",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "learning_rate": 1e-4,
    "lora_rank": 8,
    "lora_alpha": 32,
    "target_modules": "all-linear",
    "freeze_vit": true,
    "gradient_accumulation_steps": 16,
    "eval_steps": 100,
    "save_steps": 100,
    "save_total_limit": 2,
    "logging_steps": 5,
    "max_length": 2048,
    "output_dir": "output",
    "warmup_ratio": 0.05,
    "dataloader_num_workers": 4,
    "dataset_num_proc": 4,
    "model_config_override": { "circle_rope": { "alpha": 0.7, "radius": 15, "AGE_mode": "strategy_2" } }
  }
}
```
- 规则：
  - `env` 中的键值设置为进程环境变量。
  - `args` 映射为命令行参数：`{"key": "val"}` → `--key val`；列表值展开为多个位置参数。
  - 当值是对象（`dict`）时，序列化为 JSON 字符串传递（如 `model_config_override`）。
  - `custom_register_path`、`local_repo_path` 默认从脚本位置推导，允许在 `args` 中覆盖。

## 脚本改动（实现方案）
- 新增参数：`--config <json路径>`，必填；若缺失则报错并打印用法。
- 使用内联 `python` 读取并解析 JSON：
  - 读取 `env`，在 shell 中设置对应变量。
  - 将 `args` 转为安全的命令行字符串（对 `dict` 用 `json.dumps`，整体用 `shlex.quote` 防止空格与特殊字符问题）。
- 组装最终命令：
  - 固定追加 `--custom_register_path` 与 `--local_repo_path` 指向 `circle_rope/register.py` 和所在目录。
  - 打印汇总信息与最终命令，便于复现与调试。
- 保持单卡执行逻辑不变，不引入 `torchrun/deepspeed`，与当前脚本职责一致。

## 兼容性与依赖
- 不依赖 `jq`，直接使用系统 Python 的 `json` 与 `shlex`，避免额外安装。
- 不改变 `swift` 的 YAML `--config` 机制；本改造仅影响该脚本的调用方式。

## 验证方案
- 使用上面的示例 JSON，在项目根目录运行：
  - `bash circle_rope/exp/train.sh --config path/to/config.json`
- 预期：
  - 终端打印解析后的环境变量与命令参数；
  - `swift sft` 成功启动；
  - Circle-RoPE 注册成功（参考 `circle_rope/register.py`）。

## 交付内容
- 更新后的 `circle_rope/exp/train.sh` 支持 `--config` 读取 JSON 并执行训练；
- 提供一个示例 JSON（参考上面片段）供快速试跑。