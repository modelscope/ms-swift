# 训练性能采集（Profiling）

ms-swift 提供外部 callback 插件 [profiler.py](https://github.com/modelscope/ms-swift/blob/main/examples/custom/profiler.py)，用于在指定训练步采集性能数据。默认 Transformers 后端和 Megatron-SWIFT 使用相同的插件路径、callback 名称与环境变量，无需修改训练器源码。

| 训练后端 | CUDA GPU | Ascend NPU |
| --- | --- | --- |
| 默认 Transformers 后端（`swift sft/pt/rlhf`） | `torch.profiler` | `torch_npu.profiler` |
| Megatron-SWIFT（`megatron sft/pt/rlhf`） | `torch.profiler` | `torch_npu.profiler` |

插件只采集加载它的训练进程；RLHF 中独立部署的 vLLM 等 rollout 服务需要单独配置采集。Ray 等自定义启动方式也需要向实际训练进程传递插件路径和环境变量。

## 环境准备

先准备能够正常训练的环境，参考 [SWIFT 安装](../GetStarted/SWIFT-installation.md)、[Megatron-SWIFT 快速开始](../Megatron-SWIFT/Quick-start.md) 和 [NPU 支持](../BestPractices/NPU-support.md)。

- CUDA GPU：需要与 CUDA 环境匹配且支持 profiler 的 PyTorch。
- Ascend NPU：需要版本匹配的 PyTorch、`torch_npu`、CANN 和设备驱动。
- 默认训练后端不需要安装 Megatron。CUDA 采集路径不需要安装或导入 `torch_npu`。

以下示例使用 Bash，并在包含此插件的 ms-swift 仓库根目录执行。也可以把 `examples/custom/profiler.py` 复制到训练机器，使用绝对路径加载；无需复制其他插件文件。

## 快速开始

### 在现有训练脚本中启用

设置正式采集窗口：

```bash
export SWIFT_PROFILER_DEVICE=auto
export SWIFT_PROFILER_START_STEP=1000
export SWIFT_PROFILER_WARMUP_STEPS=1
export SWIFT_PROFILER_ACTIVE_STEPS=3
export SWIFT_PROFILER_OUTPUT_DIR=output/profiling
export SWIFT_PROFILER_RANKS=all
```

然后在原有 `swift sft/pt/rlhf` 或 `megatron sft/pt/rlhf` 命令中增加以下两个参数，保留原来的模型、数据、并行和训练参数：

```bash
--external_plugins examples/custom/profiler.py \
--callbacks profiler
```

这两行是需要添加的参数，不是独立的 Shell 命令。已有其他插件或 callbacks 时，将本插件和 `profiler` 追加到对应列表中。例如：

```bash
--external_plugins /path/to/my_plugin.py examples/custom/profiler.py \
--callbacks my_callback profiler
```

上述窗口在第 999 步预热，第 1000～1002 步正式采集，第 1002 步结束时导出并关闭 profiler。后续训练继续，每个进程仅采集一次。

### 默认后端短跑示例

建议先在目标机器上小步数验证加载、采集和导出，再设置正式窗口：

```bash
export SWIFT_PROFILER_START_STEP=3
export SWIFT_PROFILER_WARMUP_STEPS=1
export SWIFT_PROFILER_ACTIVE_STEPS=2

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset swift/self-cognition \
    --tuner_type lora \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_steps 6 \
    --max_length 512 \
    --split_dataset_ratio 0 \
    --save_strategy no \
    --output_dir output/profiling-sft \
    --external_plugins examples/custom/profiler.py \
    --callbacks profiler
```

第 2 步预热，第 3～4 步正式采集，训练继续到第 6 步。模型和数据集需要下载，也可以换为本地路径。设备需支持示例中的 `bfloat16`，否则按已有训练配置调整精度。

在 NPU 上使用相同的环境变量与插件参数，按已有 NPU 训练脚本设置 `ASCEND_RT_VISIBLE_DEVICES`，移除 `CUDA_VISIBLE_DEVICES`；也可显式设置 `SWIFT_PROFILER_DEVICE=npu`。Megatron 的模型加载、并行参数和环境依赖仍按 [Megatron-SWIFT 快速开始](../Megatron-SWIFT/Quick-start.md) 配置，仅追加上述两个插件参数即可。

## 配置项与默认值

所有配置都通过环境变量传入，不需要为两种后端分别增加 CLI 参数。未设置的变量使用默认值。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SWIFT_PROFILER_DEVICE` | `auto` | `auto`、`cuda`、`npu`；`cpu` 用于本地调试 |
| `SWIFT_PROFILER_START_STEP` | `1000` | 第一个正式采集步，从 1 开始，必须大于等于 1 |
| `SWIFT_PROFILER_WARMUP_STEPS` | `0` | 正式采集前的预热步数，大于等于 0 且小于 START_STEP |
| `SWIFT_PROFILER_ACTIVE_STEPS` | `1` | 连续正式采集步数，必须大于等于 1 |
| `SWIFT_PROFILER_OUTPUT_DIR` | `<训练 output_dir>/profiling` | 输出根目录；空字符串也使用此默认目录 |
| `SWIFT_PROFILER_RANKS` | `all` | 全部进程，或逗号分隔的全局 rank，如 `0`、`0,1,8,9` |
| `SWIFT_PROFILER_RECORD_SHAPES` | `true` | 记录算子输入形状 |
| `SWIFT_PROFILER_PROFILE_MEMORY` | `true` | 记录内存分配事件 |
| `SWIFT_PROFILER_WITH_STACK` | `false` | 记录调用栈 |
| `SWIFT_PROFILER_NPU_LEVEL` | `Level1` | NPU 专用，可选 `Level0`、`Level1`、`Level2` |

`auto` 优先采用训练参数中的设备；没有设备提示时，检测可用 NPU/CUDA。未找到设备时会报错，可显式设置 `cpu` 进行本地调试。插件不会改变训练所用的设备。

布尔配置接受 `true/false` 和 `1/0`，不区分大小写。整数配置不接受 `1.0`、`1.5`、`abc` 或空字符串；出错时直接终止初始化，不会悄悄改用默认值：

```text
ValueError: SWIFT_PROFILER_WARMUP_STEPS must be an integer >= 0; got 'abc'
ValueError: SWIFT_PROFILER_ACTIVE_STEPS must be an integer >= 1; got 0
ValueError: SWIFT_PROFILER_WARMUP_STEPS (10) must be smaller than SWIFT_PROFILER_START_STEP (10)
```

## Step 语义和断点续训

采集步数与训练日志中的训练步对应，不是 dataloader batch 或梯度累积 microbatch：

- 默认后端在 `on_step_begin` 读取 `state.global_step`，Megatron 读取 `state.iteration`；它们在步开始时表示已完成的训练步数。
- `START_STEP=1000` 在已完成 999 步时开始第 1000 步的正式采集。如果原 callback 的条件是 `global_step == 1000`，要保持原来的窗口，应设置 `START_STEP=1001`。
- 一个采集步覆盖该训练更新/iteration 中的全部梯度累积 microbatch。发生梯度溢出等情况时，计数遵循各训练后端本身的行为。
- 含预热的窗口在已完成 `START_STEP - WARMUP_STEPS - 1` 步时启动 profiler。
- 断点续训使用绝对步数。如果恢复时已经错过窗口起点（含预热），会打印 `skipped` 并跳过整个窗口。需要采集时，设置未来的 START_STEP。
- 如果训练在窗口前结束，会打印 `training ended before requested collection window`。应保证训练步数覆盖预热和正式采集区间。

提前结束训练时，`on_train_end` 会关闭 profiler；如果已进入正式采集阶段，底层 profiler 会尝试导出已有数据。仅完成预热时不保证产生 trace。Python 解释器正常退出时还有 `atexit` 尽力清理，强制终止、SIGKILL 或设备故障不能保证导出。

## 分布式采集

多卡和多机训练时，插件在每个选中的全局 rank 独立运行，不添加额外分布式 barrier。

```bash
# 通信分析通常需要所有 rank 的数据。
export SWIFT_PROFILER_RANKS=all

# 只分析特定进程时，可改成：
# export SWIFT_PROFILER_RANKS=0,1,8,9
```

每个节点都需要提供相同的插件路径和配置，并确保启动器把环境变量传给所有训练进程。`RANKS` 使用全局 rank，不是每台机器重复的 local_rank。输出可放在共享存储，也可以采集完成后汇总各节点的本地文件。

## 输出和分析

```text
<OUTPUT_DIR>/
  profile_step_1000/
    rank_0/
      <hostname>_<pid>_<timestamp>/
        config.json
        ... profiler 生成的文件
    rank_1/
      ...
```

rank、主机名、进程 ID 和时间戳用于避免多机或重跑覆盖。`config.json` 记录配置、训练后端、设备、全局 rank、PyTorch 版本和计划采集窗口。是否采集完整，以实际 trace 和 `stopped` 日志中的 `completed_window_steps` 为准；该计数包含预热步。

- GPU：使用 `torch.profiler.tensorboard_trace_handler` 导出，可用 TensorBoard 的 PyTorch Profiler 插件或 Perfetto 查看 `.pt.trace.json`。
- NPU：使用 `torch_npu.profiler.tensorboard_trace_handler` 导出，使用与 CANN/torch_npu 版本匹配的 MindStudio/Ascend 分析工具。Level1 提供 HCCL 通信分析相关数据，具体文件与运行环境版本有关。

GPU 和 NPU 的输出文件与分析工具不同；插件统一的是配置与 callback 生命周期。接口说明参考 [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) 和 [Ascend profiler](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000289.html)。

## 常见问题

- **没有 trace**：先确认训练命令中同时包含 `--external_plugins` 和 `--callbacks profiler`，再检查 `[SwiftProfiler]` 日志、rank 筛选、恢复的步数、输出目录及训练是否达到窗口。必须采集到正式阶段，不能仅预热。
- **提示没有预热**：默认 `WARMUP_STEPS=0` 会触发 PyTorch 的无预热提示，这是预期行为。需要预热时，设置正数并保证小于 START_STEP。
- **采集开销较高**：缩短 ACTIVE_STEPS，关闭 RECORD_SHAPES、PROFILE_MEMORY 或 WITH_STACK。内存事件采集不等于生成完整内存时间线，本插件没有额外导出内存时间线。
- **trace 混入评估或保存 checkpoint**：连续多步窗口可能包含两步之间的日志、评估和保存操作；可调整这些任务的间隔，让它们避开采集窗口。
- **已有其他 profiler**：关闭原来的 profiling callback 或其他会话，避免同一进程同时启动多个 profiler。
- **自定义 Python 启动器**：Megatron 适配依赖在正常 Megatron 入口初始化后加载插件。若先导入插件、再切换到 Megatron，需要在后端初始化完成后重新调用插件的 `register_callbacks()`。
