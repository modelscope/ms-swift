# 交接文档：dev megatron 后端下沉重构 + 遗留缺口（给无上下文的接手者）

> 本文件是**自包含**交接说明。你（qoder-cli）没有之前的对话上下文，读完这一份就能接手。
>  deeper 历史见同目录 `COMPONENT_TEST_PLAN.md`（A–G 组件测试 + SFT e2e 套件进度日志）、
> `PATCH_INVENTORY.md`（legacy→dev 补丁迁移台账）、`TRAIN_MIGRATION.md`。
> 项目规范见 `.qoder/skills/dev-module-authoring/SKILL.md`（**动手前必读**）与 `llm-terminology` skill。
> 术语：回复用标准中文 LLM 术语，禁止内部编号/代号（如 bug#4、m6gap）出现在**给用户的回复**里；
> 本文件内部为便于索引保留这些代号。

---

## 0. 一句话现状

整个大任务是「给 swift/dev（v5）SFT 全流程建端到端测试 + 补齐 examples 并真跑冒烟 + 全 feature 门禁」。
测试与 examples 已基本完成（12 个 example 真跑全绿）。**当前主线卡在最后一个门禁项**：ray 模式下
dev 的 megatron 定制整体失效（下文「主线任务」），已定位根因、已与用户敲定重构方向，**尚未动手写代码**。
另有 5 个既有小缺口待修（下文「遗留缺口」，用户已逐个给出处置决定）。

---

## 1. 大背景

- 仓库：`/mnt/data/yzhao/modelscope/ms-swift`（ms-swift v5）。分层委托架构：
  - `swift/dev/`：v5 业务编排层（config / recipe / train_loop / builders / processor / template / dataset / plugin）。**不做建模与训练内核**。
  - `twinkle/`（源码在 `twinkle/src/twinkle/`，已 editable 安装）：训练引擎本体，提供 transformers / megatron 两个后端、vLLM 采样器、DataLoader、loss/metric、CheckpointEngine 等内核。dev 把建模与训练内核**委托**给它。
  - `mcore-bridge/`：HF↔mcore 双向权重转换桥接器（`GPTBridge` / `get_mcore_model` / lora tuner 等）。
- 原始任务的 12 个测试维度：重要模型 / 模型类型（纯文本、多模态、asr、embedding、多模态 emb、reranker、generative-reranker、seq_cls）/ 框架（unsloth、transformers、megatron、liger、sentence_transformers）/ 训练方式 / 切分方式（sp、cp、dp、tp、pp、单卡）/ 评测（generate-with-predict true/false）/ sampler（vllm、transformers）/ 启动方式（ray、torchrun、单卡）/ optimizer×tuner / legacy 其他特性 / 与 legacy 的 loss 完全还原 + grid 还原。
- 覆盖策略（用户已定）：代表性覆盖 + 关键交叉；非文本模型用 Qwen 系最小档。
- 硬纪律：端到端、**不允许 mock**、UT 通过=代码完整可用；examples 必须**真跑冒烟**；UT 绿但 example 挂 → 反思 UT 为何不完整。

---

## 2. 环境与运行铁律（**踩过的坑，务必先读**）

1. **解释器钉 `/usr/local/bin/python`（3.12）**。conda `(base)` 是 3.14，`swift` 命令会解析到 stale checkout（`/mnt/workspace/lixinyu/ms-swift`）导致诡异失败。跑任何东西前 `export PATH="/usr/local/bin:$PATH"`。
2. **跑本仓 dev 测试用 `pytest` 控制台脚本，禁用 `python -m pytest`**。原因：仓库根有 `twinkle/` 命名空间目录，`python -m pytest` 把 CWD 插到 sys.path[0] 会遮蔽真包，报 `ImportError: cannot import name 'DeviceGroup' from 'twinkle'`。或显式加 `--import-mode=importlib`。
3. **数据集/模型离线**：`export MODELSCOPE_CACHE=/mnt/workspace/.cache/modelscope/hub` + `export VLLM_USE_MODELSCOPE=True`（缓存里已有 `Qwen/Qwen2.5-0.5B-Instruct` 等）。**切勿设 `HF_HUB_OFFLINE=1`**——它会阻断未预缓存的 hub 数据集下载。
4. **megatron.core 已升到 0.19.0**（`/usr/local/bin/python -c "import megatron.core"`）。0.19 收紧了 `ModelConfig` 参数校验、移除了 `get_default_save_sharded_strategy`。本会话多个 bug 都是这次版本 bump（seam drift）引爆的。
5. **GPU**：8 卡；0 与 2 常被别人占（各 ~73GB）。空闲一般是 1/3/4/5/6/7。跑前用 `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` 确认，用 `CUDA_VISIBLE_DEVICES=` 指定空闲卡。
6. **后台作业 SIGHUP 陷阱**：直接 `nohup cmd &`（无 `wait` 保活）在工具 shell 退出时 SIGHUP 会传到进程组，torchrun（`torch.distributed.run`）重注册了 SIGHUP handler → `SignalException: got signal: 1` 早期被杀。解法：用含 `wait` 的批量驱动脚本，或**前台运行**（长 timeout）。
7. **两个 torchrun 并行会抢默认端口 29500**（`EADDRINUSE`）。解法：按首个 GPU id 派生 `MASTER_PORT`（如 `$((29000 + ${GPUS%%,*}))`）。
8. **终端 banner 噪声**：每条命令输出前有一段 DSW/挂载信息表格。过滤：`grep -vE "DSW|BMCPFS|mount_path|Welcome|Mounted|^│|^╭|^├|^╰|^ ____|^\||V  V"`。
9. **验证纪律**：写框架/正式代码阶段只用 AST 校验（`ast.parse`），不擅自跑 pytest/lint；用户明确要求补测试时才写完整端到端测试。修 bug 先判归属（产品 bug 按根因在接缝处修、不削弱断言；测试 bug 修测试）。

---

## 3. 主线任务：ray 模式 dev megatron 后端注入失效（代号 bug#4）

### 3.1 现象与精确报错

跑 `test_run_sft_megatron_end_to_end[mcore-bridge]`（ray 模式）时，ray worker 构造模型阶段抛：

```
TypeError: ModelConfig.__init__() got an unexpected keyword argument 'align_grad_reduce'
```

调用栈（worker 进程内，全是 **twinkle base** 帧，**没有 dev 的帧**）：
`twinkle/infra/__init__.py new_init → _new_init_body → init_method`
→ `twinkle/model/megatron/megatron.py:__init__`（`self.strategy = MegatronStrategy(...)`）
→ `twinkle/model/megatron/strategy/megatron.py:__init__`（`self.config = self.get_model_config(...)`）
→ `twinkle/model/megatron/strategy/megatron.py:get_model_config`（`model_config = ModelConfig(...)`）→ 崩。

**受影响的 5 个 ray 模式 megatron e2e 实例**（都在 `swift/dev/tests/feature/sft/test_e2e.py`，都经 helper `_run_megatron_sft(...)`，该 helper 写死 `DistributedConfig(backend='megatron', mode='ray', nproc_per_node=2)`）：
- `test_run_sft_megatron_end_to_end[mcore-bridge]`
- `test_run_sft_megatron_end_to_end[megatron-bridge]`（若环境无 `megatron.bridge` 包会 skip）
- `test_run_sft_megatron_two_bridges_bit_identical`（同上，需 megatron.bridge）
- `test_run_sft_megatron_ga_equivalence`
- `test_run_sft_megatron_evaluate_returns_metrics`

### 3.2 根因（**关键不变量，务必理解**）

**`@remote_class` 的 ray worker 类身份 = 装饰器闭包捕获的基类，不是实际子类。**

- `twinkle/src/twinkle/infra/__init__.py` 的 `remote_class` 装饰器：`def decorator(cls): init_method = cls.__init__; ...; cls.__init__ = new_init`。ray 分支里 `RayHelper.create_workers(cls, ...)` 的 `cls` 是**被装饰的那个基类**（这里是 twinkle `MegatronModel`，见 `twinkle/model/megatron/megatron.py` 上的 `@remote_class(execute='all')`）。
- dev 现在的做法（`swift/dev/model/megatron/model.py`）：`MegatronModel(TwinkleMegatronModel)` 子类，在 `__init__` 里用 `unittest.mock.patch('twinkle.model.megatron.megatron.MegatronStrategy', _BackendBoundStrategy)` **临时掉包** twinkle 的 strategy 类来注入 dev 的桥接后端，然后 `super().__init__()`。
- **`mock.patch` 是进程内的**：它在 driver 进程生效；但 ray worker 是**另一个进程**，且 `create_workers` 用的是 twinkle 基类 → worker 里构造的是**原版 twinkle `MegatronModel` + 原版 `MegatronStrategy`**，dev 的掉包和子类完全没参与。
- 后果：
  1. dev 的定制在 ray 模式**全部失效**：`bridge_backend` 选择（永远走 twinkle 内置 mcore 路径）、`align_grad_reduce` / `nccl_comm_warmup` / `attn_impl` 的消费。
  2. 这些本是 dev-only 的 kwarg 由 `swift/dev/builders/model.py` 转发进 `MegatronModel(**kwargs)`，在 worker 里没人消费，一路漏进 twinkle base `get_model_config` 的 `ModelConfig(...)`。megatron 0.19 收紧校验 → `align_grad_reduce` 不被认识 → `TypeError`。
     - 注意 `align_grad_reduce` 语义上是 megatron **DDP 层**（`DistributedDataParallelConfig`）的 bucket 对齐概念，**不属于** `ModelConfig`(TransformerConfig)。dev 现在在 `DevMegatronStrategy.finish_param_config` 里消费它（`if not align_grad_reduce and overlap_grad_reduce: config.grad_sync_func = None`）。下沉时务必放对层：作为 strategy 的 init 参数、在 `finish_param_config` 消费，**不要**塞进 `ModelConfig`。

### 3.3 为什么 example 冒烟没抓到 / local 与 ray 两条路径

- **local（torchrun）模式**：N 个进程各跑同一份 recipe，建模就在当前进程 → dev 子类的 `__init__` 进程内执行 → `mock.patch` 生效 → 用 `DevMegatronStrategy` + dev 桥接后端。12 个 example 的 megatron（`examples/v5/train/megatron_tp_pp.sh`）走的都是 local，所以全绿。
- **ray 模式**：如上，worker 用 twinkle 基类，dev 定制失效。只有 e2e UT 用 `mode='ray'` 才踩到。
- 这正是「UT 覆盖面 > example 覆盖面」的反向案例：example 全绿不代表 ray 路径可用。

### 3.4 用户敲定的修复方向（原话要点）

1. 桥接/权重同步逻辑**直接下沉写进 twinkle 的 megatron 层，做成两个子类**；要**尽量复用逻辑**；twinkle 是底层依赖（通用能力应归它）。
2. `align_grad_reduce` 也**下沉**，或**加一个 `__init__` 参数**。
3. 模型层有 **`apply_patch` API**（见 `twinkle/model/megatron/megatron.py` 里 `apply_patch(None, MegatronBatchedP2PGroupPatch())` 这类用法）；**如果真需要 patch，用这个一等 API，而不是 `unittest.mock.patch`**。
4. 「又不想改 twinkle」是不合理的心态——**twinkle 不合理的地方就该改**（它是本次重构的一部分）。

### 3.5 已确认的重构方案（含一处必须遵守的正确性约束）

**约束**：因为 ray worker 永远实例化被 `@remote_class` 装饰的**基类**，所以「后端选择」不能靠「dev 构造哪个 twinkle 子类」来传递（那样 worker 仍退回基类）。必须以**可序列化的 `__init__` 参数**（名字字符串）riding 进基类——即用户第 2 点的思路，同样适用于 `bridge_backend`。

**下沉到 twinkle（`twinkle/src/twinkle/model/megatron/`）**
- 把 dev 的 `BridgeBackend` 协议 + `MCoreBridgeBackend` + `MegatronBridgeBackend`（现于 `swift/dev/model/megatron/bridge/{protocol,mcore,megatron_bridge}.py`）整体移进 twinkle（建议新建 `twinkle/.../megatron/bridge/`）。
  - `BridgeBackend` 协议只有两个方法：`build_model_config(hf_config, parallel_kwargs, strategy, **kwargs)` 和 `create_model(config, model_dir, *, load_weights, move_to_gpu)`，外加 `is_multimodal` 属性。它们正好对应 twinkle `MegatronStrategy` 的 `get_model_config` / `create_megatron_model` 两个接缝。
  - `MCoreBridgeBackend.build_model_config` 本就是 twinkle base `MegatronStrategy.get_model_config` 的**逐行镜像**，移下去后 base 逻辑等价、只是改成委托 `self._backend`。默认后端 = mcore，保证既有行为不变。
- `MegatronStrategy.__init__` 新增具名参数并从 `**kwargs` **pop 掉**：`bridge_backend`（名字→后端实例，twinkle 内部解析）、`align_grad_reduce`、`nccl_comm_warmup`。**这一步同时堵死 kwarg 泄漏进 `ModelConfig` 的崩点**。`get_model_config`/`create_megatron_model` 改为委托后端；`align_grad_reduce`（`finish_param_config` 里清 `grad_sync_func`）与 `nccl_comm_warmup`（`_warmup_communicators`）逻辑从 dev `DevMegatronStrategy` 移到 twinkle base strategy。→ **「两个子类」落在 strategy/后端层**，基类 `MegatronModel` 完全复用、不动它的 `@remote_class` 身份。
- `MegatronModel.__init__` 加同名参数（`bridge_backend` / `align_grad_reduce` / `nccl_comm_warmup`）透传给 strategy。
- `attn_impl` 的 FlashAttention 版本 pin：现走 `swift.dev.naming.apply_flash_version_pin`（靠翻 `transformer_engine` 模块全局量，是**进程内副作用**，必须在真正建模的 worker 里执行）。这是唯一带 dev 命名语义的部分。按用户第 3 点，若本质是打补丁就改用 twinkle 的 `apply_patch` 承载；实现时判断它更适合做成 twinkle 参数还是 `apply_patch`。**待实现时定夺**。

**dev 侧删除/收敛**
- 删 `swift/dev/model/megatron/model.py`（`MegatronModel` 子类 + `mock.patch`）、`swift/dev/model/megatron/strategy.py`（`DevMegatronStrategy`）、`swift/dev/model/megatron/bridge/`（移去 twinkle）。
- `swift/dev/builders/model.py` 的 `build_model`（megatron 分支，函数体约在文件末 ~1040–1181）直接构造 **twinkle** `MegatronModel`：把 `backend=<实例>` 改成传名字 `bridge_backend=distributed_config.bridge_backend`；`align_grad_reduce` / `nccl_comm_warmup` 作为普通 init 参数传入（不再靠 dev 子类消费）。删除 `_resolve_bridge_backend`（约 model.py:866）。
- 相关配置字段（**保持 CLI 平价，不删**）：`swift/dev/config/distributed_config.py` 的 `bridge_backend`(默认 `'mcore-bridge'`, 字面量 `Literal['mcore-bridge','megatron-bridge']`)、`align_grad_reduce`(默认 True)、`nccl_comm_warmup`(默认 False)、`use_distributed_optimizer`(默认 True)。`swift/dev/config/validate.py` 有对 `bridge_backend=='megatron-bridge'` + `max_shard_size` 的 fail-loudly 校验，下沉后保留。

**验证顺序**：先 twinkle 下沉 + AST 校验 → 删 dev 子类 + 改 builders + AST → 重跑 **local** megatron e2e（`test_run_sft_megatron_local_save_resume`，应仍绿，证明没回退）→ 重跑 **ray** megatron e2e（上面 5 个，应转绿）。

### 3.6 设计已确认（用户 2026-10-01 澄清）

用户明确：`MegatronModel` **仍是一个类**（不拆成两个 model 子类），给它加一个 `bridge_backend` init 参数（值为 `'mcore-bridge'` / `'megatron-bridge'` 名字），**基类内部按该参数选择一个 strategy 子类来用**。即「两个子类」= strategy/后端层的两个类，基类 `MegatronModel` 靠名字参数挑。

这与 3.5 的方案完全一致，也满足 3.2 的约束（名字是可序列化 init 参数，能 riding 进 ray worker 实例化的基类；worker 在基类 `__init__` 内按名字解析出对应 strategy 子类，天然生效）。**无需再向用户确认，直接按 3.5 实现。**

### 3.7 同类债（本轮**先不动**，另立一项）

`swift/dev/model/unsloth_model.py` 也用了同样的 `unittest.mock.patch` 反模式：`with patch('twinkle...get_peft_model', self._unsloth_get_peft_model)`（unsloth 框架线）。同样有「跨不过 ray actor 边界」隐患。本轮聚焦 megatron，unsloth 之后按同一思路（下沉/`apply_patch`）单独处理。

---

## 4. 本会话已修的 3 个产品 bug（**勿回退**；都是 megatron 0.19 seam drift + local 路径）

这 3 个已由 local 模式 e2e `test_run_sft_megatron_local_save_resume` 验证通过（bit-exact：resume 权重 max|diff|=0.0 / 290 参数、loss 轨迹精确续训、mcore 优化器 distcp 齐全，`1 passed in 199s`）。**注意：其中前两处正好在 bug#4 要删/移的 dev 文件里，重构时要把等价逻辑正确带进 twinkle，别丢了修复。**

1. `swift/dev/model/megatron/model.py`：原用 `functools.partial(DevMegatronStrategy, backend=...)` 掉包，但 twinkle `megatron.py` 在建 CUDA 上下文前**无条件**调 `MegatronStrategy.apply_process_env(ddp_config)`（classmethod），partial 不代理类属性 → `AttributeError: 'functools.partial' object has no attribute 'apply_process_env'`。本会话已把 partial 改成真正子类 `_BackendBoundStrategy(DevMegatronStrategy)`（`__init__` 里 `strategy_kwargs.setdefault('backend', backend)`），保留继承的 classmethod。**（重构后此文件整体删除，逻辑归 twinkle。）**
2. `swift/dev/model/megatron/bridge/mcore.py`：`MegatronConfig.use_cpu_initialization`（默认 False，非 None）被 `builders/model.py::_megatron_model_kwargs` 转发进 `config_kwargs`，与本后端硬编码的 `use_cpu_initialization=True` 冲突 → `ModelConfig() got multiple values for keyword 'use_cpu_initialization'`。修法：构造 `ModelConfig` 前 `config_kwargs.pop('use_cpu_initialization', None)`（该后端 create_model 走 CPU build→move_to_gpu，必须 True，owns 该值）。**（重构后此文件移进 twinkle，保留这个 pop。）**
3. `twinkle/src/twinkle/model/megatron/megatron.py`（`_save_mcore_optimizer`，约 1404–1453）：megatron 0.19 移除了 `get_default_save_sharded_strategy` → 存盘 `ImportError`。修法：import 改 `TorchDistSaveShardedStrategy`，`save_strategy = TorchDistSaveShardedStrategy()`（0.19 的 `dist_checkpointing.save(sharded_strategy=None)` 内部默认就是它），带版本兼容注释；后续 `thread_count` 微调与 `FullyParallelSaveStrategyWrapper` 包装不变。**（这是 twinkle 侧修改，重构保留。）**

---

## 5. examples 与冒烟驱动现状（12/12 真跑真存盘全绿）

- `examples/v5/train/` 12 个脚本：`sft.sh / seq_cls.sh / embedding.sh / reranker.sh / liger.sh / galore.sh / muon.sh / multimodal.sh / eval_generate.sh / torchrun_dp_sp.sh / ray.sh / megatron_tp_pp.sh`；本地样本数据在 `examples/v5/train/data/`（含 `vl.jsonl`，多模态用，行格式 `{"messages":[{"role":"user","content":"<image>..."},{"role":"assistant","content":"..."}], "images":["http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/{cat,animal,ocr}.png"]}`）。
  - `multimodal.sh` 用本地 `data/vl.jsonl`，**不要**用 `coco-en-mini`（dev 注册的 `modelscope/coco_2014_caption` 是脚本式数据集，新 `datasets` 报 `Dataset scripts are no longer supported`）。
- 冒烟驱动（临时工具，非产品）：`.scratch_offload/smoke_one.sh`（单条：GPU 重映射 + 步数上限 + 环境修正）、`.scratch_offload/smoke_batch.sh`（批量并行，末尾 `wait` 保活）。`smoke_one.sh` 关键约定：
  - `export PATH="/usr/local/bin:$PATH"`；`MODELSCOPE_CACHE` / `VLLM_USE_MODELSCOPE`；`MASTER_PORT="$((29000 + ${GPUS%%,*}))"`；**无 `HF_HUB_OFFLINE`**。
  - dev CLI 的 `normalize_argv` **拒绝重复冲突 flag**（非 legacy 的 last-wins，会报 `Conflicting values for --X`）。所以注入步数上限时用 sed **就地替换**已存在的 `--save_steps`/`--output_dir`/`--eval_steps`，只**追加**不预存在的 `--max_steps`(transformers) / `--train_iters`(megatron，经 process.py 映射到 max_steps)。
- v5 启动约定：`USE_SWIFT_V5=1 swift sft ...` 路由到 `swift.dev.cli.sft`（`/usr/local/bin/swift`，shebang `#!/usr/local/bin/python`）。v5 无独立 megatron 命令，用 `swift sft --backend megatron`。`NPROC_PER_NODE=N` env 触发 cli 自动包 `torch.distributed.run`（local 模式）；ray 走 `--mode ray --nproc_per_node N`。

---

## 6. 遗留缺口清单 + 用户处置决定（本轮已授权）

都是「legacy 支持、dev 迁移时漏接线」的参数/能力：CLI 能解析、config 能存，但下游没人用或用了出错。按项目规范「死参数不许静默失效」处理。

| 代号 | 内容 | 用户决定 |
|---|---|---|
| m6gap | dev LoRA + seq_cls/reranker 时，新加的**分类头未入 `modules_to_save`** → 存 checkpoint 可能漏存头、加载回来是随机头 | ✅ **已修复** |
| asrgap | twinkle **音频 collate 与 Qwen2.5-Omni 不兼容**，音频输入喂不进（现测试标 xfail，见 `test_multimodal.py` 的 `_OMNI_AUDIO_XFAIL`） | ✅ **已修复** |
| m7gap1 | `neftune_noise_alpha` **全仓无消费者**（静默失效） | **暂不修**，先标注「不起作用」 |
| m7gap2 | `full_determinism` + `seed` **未透传 `twinkle.initialize`**（设了不一定真复现） | ✅ **已修复** |
| m7gap3 | `freeze_parameters` / `trainable_parameters`（+ratio/regex）**无消费者**（设了不生效） | ✅ **已修复** |

**建议推进顺序**：先做主线 bug#4 重构（挡着 5 个 ray megatron e2e 门禁）→ 跑绿 → 再依次 m7gap2 / m7gap3 / m6gap / asrgap（m7gap1 只加标注）。每修一个先判归属、按根因在接缝处修、不削弱断言，修完重跑对应模块，最后整体门禁。

**进度**：m7gap2、m7gap3、m6gap、asrgap 均已完成；剩 m7gap1 只加标注（桩已在 `test_legacy_features.py`，本轮核对）。落地记录见下方各段末尾。之后跑最终整体门禁（§7）。

### 6.1 缺口的精确落点（已核实，行号会漂移，以符号名为准）

**m7gap2 — seed / full_determinism 未透传 `twinkle.initialize`**
- 配置源：`swift/dev/config/train_config.py` 的 `seed: int = 42`、`full_determinism: bool = False`。
- 断点：`swift/dev/recipe/assembly.py::TrainAssembly.initialize_twinkle(distributed_config)`——签名只收 `distributed_config`，ray 分支 `twinkle.initialize(mode='ray', nproc_per_node=..., name=..., groups=[...])` 与 local 分支 `twinkle.initialize(mode='local')` **都没传 seed / full_determinism**。
- twinkle 侧本就支持：`twinkle/src/twinkle/infra/__init__.py::initialize(mode, nproc_per_node, ncpu_proc_per_node, seed=42, full_determinism=False, groups, ...)`，内部 `framework_util.seed_everything(seed, full_determinism)`——只是 dev 没喂。
- 调用方（全部只传 distributed_config，都要改）：`run_sft.py` / `run_dpo.py` / `run_seq_cls.py` / `run_embedding.py` / `run_reranker.py` / `run_gkd.py` / `run_infer.py` 里的 `TrainAssembly.initialize_twinkle(distributed_config)`；RL 另有 `run_grpo.py::_initialize_twinkle_rl`（run_ppo.py 复用它）。
- 修法：`initialize_twinkle` 增参（收 `train_config`，或直接收 `seed`+`full_determinism`），两分支都传 `seed=train_config.seed, full_determinism=train_config.full_determinism`；改所有调用方把 train_config 串进去。`run_infer` 若无 train_config → 用默认值。注意 seed 还影响 megatron 建模 RNG（`model_parallel_cuda_manual_seed`），是可复现性承重点，别只在 transformers 侧生效。

**m7gap3 — freeze_parameters / trainable_parameters 无消费者**
- 配置源：`swift/dev/config/adapter_config.py` 的 `freeze_parameters`(list) / `freeze_parameters_regex` / `freeze_parameters_ratio`(0~1) / `trainable_parameters`(list) / `trainable_parameters_regex`；`TunerConfig` 定义在同文件顶部。
- 现状：**全 dev 无消费者**（grep `requires_grad`/`freeze_parameters`/`activate_parameters` 只命中 `swift/dev/model/loader/qwen.py` 里 Qwen3-TTS 的专用冻结，无关）。用户设了不生效。
- legacy 目标行为（见 `TRAIN_MIGRATION.md` 的 full/freeze 段与 full 分支段）：full 训练时 `model.train()` + `requires_grad_(True)` + `freeze_parameters(ratio/list/regex)` + `activate_parameters(trainable_*)`，优先级 **trainable_* > freeze_***；多模态另有 `freeze_llm/vit/aligner`。legacy swift 有 `freeze_parameters` / `activate_parameters` helper 可参照语义。
- 修法：在 dev 的**全参训练分支**消费这些字段（先冻结 list/regex/ratio，再解冻 trainable_*）。落点需先定位——`swift/dev/builders/model.py` 是唯一含 tuner/train_type 装配的 builder，找到 `train_type=='full'` 的模型准备处接入。megatron 侧参照 `TRAIN_MIGRATION.md` 的 MegatronTunerMixin 段（注意 `freeze_parameters_ratio` 与 PP>1 互斥，需 fail-loudly raise）。
- ✅ **已落地**（本会话）：
  - 5 个字段从 `TunerConfig`（`adapter_config.py`）**移到 `TrainConfig`**（`train_config.py` 的 Training Strategy 段）。原因：full 跑不带 `TunerConfig`（`select_tuner` 与 `_select_megatron_tuner` 都把 `tuner='full'` 映射成 None），字段留在 TunerConfig 会在最需要它的分支被丢弃——与 GaLore 旋钮放 TrainConfig 同理。`freeze_llm/vit/aligner` 属多模态 target 选择，**仍留在 TunerConfig**（不在本缺口范围）。
  - twinkle 新增接缝：`twinkle/model/base.py` 的纯函数 `_freeze_then_activate`（ratio 按累积元素数 cumsum+bisect、name 前缀、regex search；freeze_* 先、trainable_* 后 → **trainable 胜**；坏 regex 直接 raise 而非 legacy 的 warn-and-skip）+ `TrainableModel.freeze_parameters`（`@remote_function(dispatch='all', collect='none', lazy_collect=False)`，unwrap 后遍历 named_parameters 委托纯函数；**不**做 legacy 的 blanket `requires_grad_(True)`，以保留 loader 级冻结如 Qwen3-TTS speaker_encoder）。
  - dev 消费点：`builders/model.py::apply_full_param_freeze(model, train_config)`（无旋钮则 no-op，避免无谓 remote round-trip），在 `recipe/assembly.py::build_model` 的 **full 分支**（`else` of `tuner_config is not None`）调用，位置在 `configure_optimizer` 之前（optimizer 按 requires_grad 建 param group）。
  - 守卫：`config/validate.py::_check_freeze_ratio_pp`（config-only）——full + megatron + `freeze_parameters_ratio>0` + `PP>1` → fail-loudly raise。
  - 测试：`tests/component/optimizer/test_freeze_parameters.py`（16 项，纯逻辑+转发+守卫，GPU-free）；`tests/feature/sft/test_optim_tuner.py::test_full_param_freeze_holds_frozen_weights`（真权重 e2e：冻结层 bit-identical、未冻结层移动）。旧的 `test_freeze_and_trainable_parameters_are_unwired_gaps` 桩已删。零回归（既有失败经 stash 复核均为改动前既有）。

**m6gap — LoRA + seq_cls/reranker 分类头未入 modules_to_save**
- 头的构建：`swift/dev/builders/model.py::_apply_seq_cls_head`（把 seq_cls/reranker 路由到 num_labels 宽的 HF SequenceClassification 头，设 `config.num_labels` 等；transformers 路径调用点在 build_model 内）；megatron 侧头在 build_model 的 megatron 分支（reranker→num_labels=1，seq_cls→num_labels）。
- LoRA 配置装配：`swift/dev/adapter.py::_lora_common_kwargs(cfg)` 里 `modules_to_save=(cfg.modules_to_save or None)`——**只透传用户显式给的**，不自动加分类头；同文件 `_build_adapter_config(cfg, ...)` 注释明写 “task_type is intentionally NOT set”，即当前 adapter 构建**不知道 task_type**。
- 缺口：LoRA + seq_cls/reranker 时，分类头（HF SequenceClassification 头模块名多数模型是 `score`，部分是 `classifier`）不在 modules_to_save → 头不被 LoRA 训练/保存，存出的 checkpoint 加载回来是**随机初始化头**。
- legacy 参考：`TRAIN_MIGRATION.md` 的 `get_modules_to_save` 段——legacy 会给 seq_cls(reward) 追加 `v_head`，并展开 `all-embedding`/`all-norm`。
- 修法：把 task_type（或已构建模型的头模块名）串进 `_build_adapter_config` / `_lora_common_kwargs`，当 task_type ∈ {seq_cls, reranker} 时把头模块名 append 进 modules_to_save。头模块名**最好从已构建的模型探测**（避免硬编码 score/classifier 的模型间差异），或沿用 legacy 约定。修完需端到端验证：LoRA 训 seq_cls → 存 → 重新加载 → 头权重非随机且可复现预测。
- ✅ **已落地**（本会话）：
  - `swift/dev/adapter.py`：新增两个模块常量 `_SEQ_CLS_HEAD_MODULES = ('score', 'classifier')`（HF `*ForSequenceClassification` 头的家族相关名——Qwen/LLaMA 系用 `score`，Bert/Deberta 系用 `classifier`）与 `_HEAD_SAVING_TASK_TYPES = ('seq_cls', 'reranker')`（`generative_reranker` 故意排除：它保留 CausalLM 词表头、无新模块；PPO 价值头走 `task_type='seq_cls'` 一并覆盖）。
  - **不从已构建模型探测头名**：Ray 下 `apply_tuner` 在 driver 对**代理模型**执行，peft config（含 modules_to_save）先序列化再发给 worker 建 adapter，driver 侧探不到真实模块。改为**两个候选名都列**——peft 的 `_set_trainable` 按 `name.endswith(entry)` 匹配、`strict_module_check=False` 静默忽略未命中的那个，所以列两个是安全的：家族真有的被包进 modules_to_save，另一个 no-op。
  - 串线：`_lora_common_kwargs(cfg, task_type=None)`（**copy** `cfg.modules_to_save` 而非原地改，复用的 TunerConfig 保持用户原值；`task_type ∈ _HEAD_SAVING_TASK_TYPES` 时追加缺失的头名）← `_build_adapter_config(cfg, *, num_training_steps=None, task_type=None)`（lora 与 adalora 两路都转发）← `apply_tuner(..., task_type=None)` ← `recipe/assembly.py::build_model` 传 `task_type=self.task_type`。adalora 是 LoraConfig 子类、同样携带 modules_to_save，故一并生效。
  - 注意：`_build_adapter_config` 里“task_type is intentionally NOT set”指的是 **peft LoraConfig 自身的 `task_type` 字段**（设了会得到 PeftModelForCausalLM，其 forward 读 `base_model.config.model_type`，而 Megatron 的 config 是 mcore `ModelConfig` 只有 `hf_model_type` → AttributeError）。本次新增的 `task_type` **入参**是 dev 的 task_type，只用于决定头是否入 modules_to_save，与该 peft 字段无关；docstring 已改词消歧。
  - 测试：`tests/component/model/test_adapter_modules_to_save.py`（12 项，GPU-free：seq_cls/reranker 追加、causal_lm/generative_reranker 不加、用户 modules_to_save 保留且去重、cfg 不被 mutate、adalora 也带头、apply_tuner 转发）；`tests/feature/sft/test_task_types.py::test_run_seq_cls_lora_saves_classification_head`（真权重 e2e：LoRA seq_cls 训→存，断言 checkpoint-final 里存在完整头张量 `...score.weight`（`endswith` 只命中头、不命中 lora_A/lora_B）且有限非零）。**非空洞验证**：临时把 `_HEAD_SAVING_TASK_TYPES=()` 关掉修复后重跑，`score.weight` 从 checkpoint 消失（只剩 lora 分片），确认断言真的钉住了修复。回归：`test_task_types.py` 全 6 项 + `test_optim_tuner.py::test_adamw_trains_full_and_lora[full/lora]` 均绿。
  - 作用域：transformers 路径。Megatron 侧头名不同（`output_layer`），但 peft 忽略未命中项，故对 megatron 跑 seq_cls/reranker 也无害；megatron 的 LoRA 头处理若另有缺口需单独定位（本轮未触及）。

**asrgap — twinkle 音频 collate 与 Qwen2.5-Omni 不兼容**
- 归属：**twinkle**（`twinkle/src/twinkle/processor/base.py::_collate_macro_batch`）。twinkle 的音频 collate 只硬编码了**一种**契约（gemma4 的 channels-last），dev 的 `InputProcessor` 未覆写该方法、`collate_mm_data` 钩子在产线也从未接线（`processor._template` 恒为 None），所以根因只能在 twinkle 修。
- 两种契约共用字段名 `input_features`，靠** accompanying mask 字段名**区分：
  - gemma4（mcore-bridge `mm_gpts/gemma4.py`）：`input_features` + `input_features_mask`；每样本 2-D `[plane0,plane1]` → `unsqueeze(-1)` 补尾轴成 channels-last，1-D mask expand 成特征平面，batch 折进 dim 0。
  - Qwen2.5-Omni / Qwen-audio：`input_features` + `feature_attention_mask`；transformers `Qwen2_5OmniThinker.get_audio_features` 要每音频成批的 `[N,freq,time]` 特征 + `[N,time]` mask，纯 dim-0 concat，**不 reshape**。
- 旧 bug：twinkle 对 `input_features` **无条件**套 gemma4 reshape（只认字段名），且开头 blanket `.squeeze()` 把 Omni 的前导音频维剥掉（`[1,128,T]→[128,T]`），concat 后成 `[128*B,T,1]`，`get_audio_features` 里 `permute(0,2,1)[mask.bool()]` 对 `[256,1,30000]` vs mask `[2,30000]` → `IndexError`。**实测**（临时在 collate 打点 + `--runxfail` 跑 omni e2e）确认每行到达 collate 时张量完好为 `[1,128,30000]` / `[1,30000]`（arrow round-trip 未塌维）。
- 修法（twinkle，两处）：collate 开头按 `qwen_audio = any('feature_attention_mask' in inp ...)` 判定契约；(1) blanket squeeze 对 `qwen_audio` 批次的 `_QWEN_AUDIO_FIELDS=('input_features','feature_attention_mask')` **跳过**（保留前导音频维）；(2) VLM 分支的 gemma4 assert/reshape 用 `if not qwen_audio:` 包住，Qwen-audio 直接原样 dim-0 concat。**gemma4 路径逐字节不变**（`qwen_audio=False` 时 squeeze 与 reshape 与旧代码完全一致）。
- 测试：`tests/component/processor/test_audio_collate.py`（3 项，GPU-free，直接喂 CPU 张量、audio-only 行、batch=2 调 `collate_fn`）——Qwen 契约保 `[B,freq,time]`/`[B,time]` 且值不变、多音频行保前导维不触发 gemma4 的 2-D assert、gemma4 契约保 `[2*p0,p1,1]`/`[2*p0,p1]` 逐值一致；`tests/feature/sft/test_multimodal.py::test_run_sft_omni_audio_end_to_end`（真权重 e2e，已**去掉 xfail**，loss 12.36→11.26 有限收敛，证明音频特征真被消费）。xfail 常量 `_OMNI_AUDIO_XFAIL` 已删，模块注释改为 FIXED 说明。
- 回归复核：`test_packing.py` 有 2 项失败（`test_packing_derives_padding_free`、`test_packed_collate_produces_packed_position_ids_and_cu_seqlens`），经 `git stash` twinkle 改动后重跑**仍失败** → 改动前既有（`_get_packed_seq_params` 签名漂移 + padding_free 派生），与本音频修复无关。

---

## 7. 门禁状态（M9-d：全 feature 目录）

- 已绿（本会话验证）：
  - `test_run_sft_megatron_local_save_resume`（local 模式 megatron e2e）→ `1 passed`，验证第 4 节 3 个修复。
  - 之前里程碑：`test_task_types / test_multimodal / test_frameworks / test_distributed / test_eval_sampler / test_optim_tuner / test_legacy_features / test_parity_grid` 逐模块门禁通过（详见 `COMPONENT_TEST_PLAN.md` 进度日志）。
- 红（待 bug#4 重构修复）：第 3.1 节列的 5 个 ray 模式 megatron e2e。
- **尚未整体跑一遍** `swift/dev/tests/feature`（含 grpo/rl/sft 全部 slow）作为最终门禁——等 bug#4 修完再整体跑。

### 运行命令模板

```bash
cd /mnt/data/yzhao/modelscope/ms-swift
export PATH="/usr/local/bin:$PATH"
export MODELSCOPE_CACHE=/mnt/workspace/.cache/modelscope/hub
export VLLM_USE_MODELSCOPE=True

# 单个 megatron e2e（local，应绿）——前台跑，避免 SIGHUP
CUDA_VISIBLE_DEVICES=1,3 pytest swift/dev/tests/feature/sft/test_e2e.py -m slow \
  -k "test_run_sft_megatron_local_save_resume" \
  --import-mode=importlib -p no:cacheprovider -s 2>&1 | tee /tmp/meg_local.log

# ray 模式 megatron e2e（当前红，重构后应转绿）
CUDA_VISIBLE_DEVICES=1,3 pytest swift/dev/tests/feature/sft/test_e2e.py -m slow \
  -k "test_run_sft_megatron_end_to_end and mcore" \
  --import-mode=importlib -p no:cacheprovider -s 2>&1 | tee /tmp/meg_ray.log

# 最终整体门禁（bug#4 修完后）
CUDA_VISIBLE_DEVICES=<空闲卡> pytest swift/dev/tests/feature -m slow \
  --import-mode=importlib -p no:cacheprovider 2>&1 | tee /tmp/feature_gate.log
# 看末尾 "N passed/failed" 与 EXIT；teardown 期 EngineDeadError/graceful_shutdown/raylet zombie 是关停噪声，不计失败
```

---

## 8. 关键文件索引

**dev megatron（bug#4 重构主战场）**
- `swift/dev/model/megatron/model.py` — dev `MegatronModel` 子类 + `mock.patch` 掉包（**待删**）。
- `swift/dev/model/megatron/strategy.py` — `DevMegatronStrategy`：`get_model_config`/`create_megatron_model` 委托后端；`__init__` 消费 `backend`/`attn_impl`/`align_grad_reduce`/`nccl_comm_warmup`；`finish_param_config` 用 align_grad_reduce 清 grad_sync_func；`_warmup_communicators`（**逻辑待移进 twinkle，文件待删**）。
- `swift/dev/model/megatron/bridge/{__init__,protocol,mcore,megatron_bridge}.py` — 桥接后端抽象 + 两实现（**待移进 twinkle**）。`megatron_bridge.py` 里 `_MCoreCompatBridgeShim` 把 NVIDIA AutoBridge 的 `load/save/export_hf_weights` 适配成 twinkle 期望的 mcore 兼容接口。
- `swift/dev/builders/model.py` — `build_model` megatron 分支（约末段 1040–1181）组装 `MegatronModel(**model_kwargs)`；`_megatron_model_kwargs`（转发 MegatronConfig 非 None 字段，注意 `_NON_MODEL_MEGATRON_FIELDS` 排除表**不含** use_cpu_initialization）；`_resolve_bridge_backend`（约 866，**待删**）。
- `swift/dev/config/distributed_config.py` — `bridge_backend`/`align_grad_reduce`/`nccl_comm_warmup`/`use_distributed_optimizer` 等字段（**保留**）。
- `swift/dev/config/validate.py` — megatron-bridge + max_shard_size 的 fail-loudly 校验等（**保留**）。

**twinkle megatron（下沉目标）**
- `twinkle/src/twinkle/model/megatron/megatron.py` — `MegatronModel`（`@remote_class(execute='all')`）；`__init__` 里 line ~128 `MegatronStrategy.apply_process_env(ddp_config)`（classmethod，CUDA 上下文前调）、line ~176 `self.strategy = MegatronStrategy(...)`；`_save_mcore_optimizer`（约 1404–1453，第 4 节 bug#3 已修）。
- `twinkle/src/twinkle/model/megatron/strategy/megatron.py` — `MegatronStrategy`：`__init__`（约 98–179，`**kwargs` 会流到 get_model_config）、`apply_process_env`（classmethod）、`get_model_config`（约 428–462，`ModelConfig(...)` 崩点）、`create_megatron_model`、`finish_param_config`、`finalize_model_grads_for_lora`。
- `twinkle/src/twinkle/infra/__init__.py` — `remote_class` 装饰器（约 786–983，`create_workers(cls)` 在 ~949，`cls.__init__ = new_init` 在 ~980）、`remote_function`（约 986+，`_mode=='local'` 时直调）、`get_device_mesh()` 公开访问器、`apply_patch` 一等补丁 API。

**dev 里同类模式参考（正确范例）**
- `swift/dev/model/sentence_transformer_model.py` — `SentenceTransformerModel(TwinkleTransformersModel)`：**完全重写 `__init__`、不调 base 被 `@remote_class` 包装的 `__init__`**，自己 `_try_init_process_group()` + `self.device_mesh = device_mesh if device_mesh is not None else get_device_mesh()`。这是 dev 里「子类化 @remote_class 模型」的既有成熟做法（但注意：它对 megatron 未必是最优解，因为 megatron 要跨 ray worker，用户要求的是**下沉 twinkle**而非在 dev 重写）。

**测试**
- `swift/dev/tests/feature/sft/test_e2e.py` — megatron e2e（local + ray）；helper `_run_megatron_sft`（ray，约 405–460）、`_run_backend_subprocess`（local torchrun 子进程）、`_write_toy_dataset`；`MODEL='Qwen/Qwen2.5-0.5B-Instruct'`。
- `swift/dev/tests/feature/sft/test_multimodal.py` — VL 用本地 hermetic jsonl（`_write_vl_sft_data`）+ tiny 随机模型，从不走 coco hub；audio(Qwen2.5-Omni) 标 xfail（asrgap）。

---

## 9. 下一步（接手后立即要做）

1. 读 `.qoder/skills/dev-module-authoring/SKILL.md`（分层铁律 / 放置复用 / 失败语义 / 测试纪律 / 重构工作流清单）。
2. 精读 twinkle `MegatronModel` + `MegatronStrategy` + `infra.remote_class`/`apply_patch` 全貌，以及 dev 三个 bridge 文件与 `builders/model.py` megatron 分支。
3. 按第 3.5 节方案实现下沉重构（先 twinkle、AST 校验；再删 dev 子类 + 改 builders、AST）。核对运行期契约（`ModelConfig`/`TransformerConfig` 字段名、`apply_patch` 用法、导出符号、import 路径——AST 查不出这些）。
4. 重跑 local（不回退）+ ray（转绿）megatron e2e；再整体跑 `swift/dev/tests/feature` 门禁。
5. 依次修 m7gap2 / m7gap3 / m6gap / asrgap，m7gap1 只加标注。
6. 全部绿后，考虑把「@remote_class worker 只认基类 → driver 端 mock.patch/子类定制在 ray 模式失效」这条教训固化进 skill/记忆（若尚未），并反思 megatron e2e UT 为何在 seam drift 后没被重跑（→ 重 e2e 须定期真跑）。
