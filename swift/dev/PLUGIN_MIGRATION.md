# 插件机制最终形态（统一加载器 + kind 清单）

> 记录插件机制在 dev 的最终形态、结论与依据。与 `MODEL_MIGRATION.md`/`DATASET_MIGRATION.md` 同体例：**只记已经落地的**，未落地的写清卡在哪，不留悬空。

> 一句话结论：**通用加载能力下沉 twinkle**（`twinkle.utils.load_module` 支持 本地文件 | 本地文件夹 | hub id，路径派生唯一模块名，不再串号）；**扩展点契约留在 swift**（`swift/dev/plugin.py::PluginRegistry` 只声明 kind + 基类，加载委托 twinkle）；每个 kind 的命令行统一成 `--<kind> 名字|类名|id`，外部实现经唯一的 `--external_plugins`（本地文件 | 本地文件夹 | hub id）注册，且必须是该 kind 基类的子类；奖励选择器合并为 `--orm`/`--prm`；6 种 legacy kind 已彻底移除。

## 判定规则
- **已接入（wired）**：dev 有 `PluginKind` 声明 + Config 字段 + 消费方。
- **归内核（kernel）**：这类"插件"在 dev 里不是扩展点，而是 twinkle 的内核构件（loss / optim / scheduler / metric 由 twinkle 提供实现，dev 只做名字映射，经 `naming.py::resolve_*`）。
- **不是扩展点（n/a）**：dev 明确不由插件表达（tuner 是能力映射器，不是名字注册表）。
- **已移除（removed）**：legacy 曾有的 kind，在 dev 彻底删除，`--xxx` 由 `legacy_coverage.py` 早失败并给替代方案。

---

# 一、统一加载器（twinkle）

`twinkle/src/twinkle/utils/loader.py`：

- `load_module(source)` 是**单一加载入口**，`source` 支持三来源：本地 `.py` 文件、本地文件夹（`sys.path.insert` 后导入其 `__init__.py`）、`hf://`/`ms://` hub id（缺则下载，维持 `trust_remote_code` 门）。加载按**路径派生的唯一模块名**（`<sha1(abspath)[:8]>_<stem>`）缓存进 `sys.modules`，两个都叫 `plugin.py`/`__init__.py` 的文件不再互相覆盖，加载幂等。
- `Plugin.load_plugin(source, base)` 复用 `load_module`，返回该模块里 `base` 的子类；`construct_class(spec, base, module)` 是"实例直返 / 子类构造 / 字符串先在命名空间查、查不到回落 `load_plugin`"的 name|class|id 解析器。
- **信任门决策**：本地文件/文件夹是用户在自己机器上显式传入的，直接加载、不加 hub 的 `trust_remote_code` 门；hub id 维持 safe-mode 拒绝。

swift 侧不再自带 `spec_from_file_location` 实现：`PluginRegistry.load_external(paths)`（`swift/dev/plugin.py`）逐项委托 `twinkle.utils.load_module`，只保留"先 `_declare_builtin_kinds()` 再导入用户文件"的顺序保证（`_BUILTIN_KIND_MODULES = ('swift.dev.rewards.orm', 'swift.dev.rollout.sandbox')` 惰性 importlib 导入声明 kind，修过的导入顺序 bug 不回退）。

# 二、统一名字解析层（swift）

`swift/dev/naming.py::resolve_plugin_class(spec, base, registry=None, *, kind, resolve_id=None) -> Type` 是所有扩展点共用的"名字/类/id/外部源 → 实现类"解析器，语义与 twinkle `construct_class` 一致：

- `spec` 是类 → 校验 `issubclass(base)` 后返回；
- 是字符串且命中 `registry` → 返回注册类；
- 命中 `resolve_id`（该 kind 自己的 id/path 解析，如 model family loader、dataset loader）→ 返回其结果；
- 其余字符串 → 当外部插件源，经 `Plugin.load_plugin(spec, base)` 加载。

**只返回类不构造**：各 kind 的构造参数差异（reward 的 `cls(args=config)`、sampler 的 engine kwargs、loader 的 dataset info）留在调用点，不强求统一签名。内核类（loss/optim/scheduler/strategy）走同文件的 `resolve_loss`/`resolve_optim`/`resolve_scheduler`/`resolve_strategy` 与统一入口 `resolve(category, name)`。

# 三、kind 清单（最终形态）

| kind / 扩展点 | 基类 | 注册表 | CLI | 结论 |
|---|---|---|---|---|
| reward（ORM 规则） | `RewardPlugin` | `swift/dev/rewards/orm.py::orms`（`register_kind('reward', ..., config_field='orm')`） | `--orm 名字\|类名\|模型 id\|外部源` | **已接入** |
| reward（PRM 规则） | 同上（同签名，不另开 kind） | `prms` | `--prm ...` | **已接入**（同一 kind） |
| tool | `ToolPlugin` | `swift/dev/rollout/sandbox.py`（`config_field='tools'`） | `--tools 名字` | **已接入** |
| sampler | twinkle `Sampler` | `swift/dev/builders/sampler.py::_SAMPLERS`（`_sampler_kind()` 惰性声明，`config_field='sampler'`） | `--sampler vllm\|sglang\|transformers\|pt\|名字\|类名\|外部源` | **已接入** |
| model | `ModelLoader`（twinkle `ModelLoaderProtocol`） | `swift/dev/model/loader/base.py::MODEL_MAPPING` | `--model id\|路径`，`--model_type 注册名\|外部类名` | **已接入**（`_resolve_model_loader` 经 `resolve_plugin_class`） |
| dataset | `DatasetLoader` | `swift/dev/dataset/loader/base.py::DATASET_MAPPING` | `--dataset id\|文件\|目录\|注册名\|外部类名` | **已接入**（`match_dataset_type` 认注册名） |
| template | twinkle `Template` | legacy `get_template` 注册表 | `--template 名字` | **已接入（间接）**：`builders/template.py` 仍走 legacy `get_template` + `shifted_template_class` 派生；外部模板经 `--external_plugins` 导入后由 legacy 注册表解析 |
| loss_scale | legacy loss_scale 注册表 | — | `--loss_scale 名字` | **已接入（间接）**：透传给 legacy `get_template`，dev 不重复造 |
| loss / optim / scheduler / metric | twinkle 内核基类 | twinkle `torch_loss_mapping` 等 | `--loss` / `--optim` / `--lr_scheduler` | **归内核**：dev 只做名字映射，外部名经 `resolve_*` → `Plugin.load_plugin` 接入 |
| tuner | peft tuner config | — | `--tuner lora\|adalora\|trainable_tokens\|full` | **不是扩展点**：`adapter.py::apply_tuner` 是能力映射器（字符串→peft Config），加一种 tuner 是加实现不是注册名字 |

## sampler kind 为什么惰性声明

`import twinkle.sampler` 会连带拉入 vllm + torch。若把 sampler 加进 `_BUILTIN_KIND_MODULES`，每次 `--external_plugins` 加载（含纯 SFT / reward-only 运行）都会被强制导入 vllm，是回归。因此 sampler kind 在 `_sampler_kind()` 里惰性声明——只在 `build_sampler` 内部调用，此时引擎栈无论如何都要导入。外部 sampler 经 `--sampler <源/类>` 直接由 `Plugin.load_plugin` 回落解析，无需预声明；命名式外部 sampler（`@register('sampler', name)`）需在 `load_external` 前声明 kind，这一局限记在 `builders/sampler.py` docstring。

# 四、奖励选择器合并为 --orm / --prm

`--orm` 与 `--prm` 各接一个**异构列表**，每项是 规则名 | 模型 id | 类名 | 外部插件名；判别用"注册名优先"（命中 `orms`/`prms` → 规则函数；否则 → 模型 id 走 `_resolve_reward_model`）。字段统一收到 `RLHFConfig`（`orm`/`orm_weights`/`prm`/`prm_weights`），`run_infer` 经可选 `rlhf_config` 读取。奖励模型专属子旋钮（`orm_adapter`/`prm_adapter`、`reward_template`/`judge_template`、`normalize_rewards`）保留在 `InferConfig`——合并的是选择器，不是模型分支的专属 config。

> **每通道并行布局已接线**：`RLHFConfig.orm_parallel_spec`/`prm_parallel_spec`（`Optional[str]`，紧邻各自的 `orm`/`prm`）是「按模型 id 传入的该通道奖励模型」自己的布局，区别于 run 的 `--parallel_spec`（policy/sampler 的）。`None` 时该通道沿用 `nproc_per_node` 宽、纯 DP（旧行为）。有值时 `_RewardModelSpec.parallel_spec` 一路驱动：该通道独占 `DeviceGroup` 的宽度（`world_size`）与每 worker 可见卡数（`gpus_per_worker`＝spec 的 tp 维）、标量 RM 的 `DeviceMesh`（`build_frozen_reward_model`→`build_model`）、生成式 judge 的 DP mesh 与引擎 `tensor_parallel_size`。标量 RM 走 transformers 后端，只按 dp/fsdp/ep/ulysses 切权重，tp/pp 不是它的权重切分维——要 tp/pp 需 megatron 后端或走生成式 judge。

> `RLHFConfig.loss_type`（GRPO 目标，`Optional[List[str]]`）**保持原名**，与 `TrainConfig.loss`（监督损失，由 `loss_type` 改名）是两回事。`legacy_coverage.py` 的 owner-aware 别名保证 `--loss_type` 在 rlhf 命令下仍直指 `RLHFConfig.loss_type`、在 pt/sft 下才别名到 `TrainConfig.loss`（`normalize_argv` 与 `build_legacy_contract` 都跳过"源名本身是另一个 Config 现存字段"的跨 owner 别名）。

# 五、已移除的 6 种 legacy kind

| legacy kind | dev 处置 |
|---|---|
| `rm_plugins` | **移除**：奖励模型打分一律走 dev 原生 `_ScalarRewardModelPlugin`（scalar）/生成式 judge，`build_reward_model_plugins` 不再查 legacy dict；`RLHFConfig.reward_model_plugin` 字段删除 |
| `callbacks` | **移除**：`TrainConfig.callbacks` 无消费方，删除；`recipe/tracking.py` 里 swanlab 的局部 `callbacks` 与此无关，保留 |
| `agent_template` | **移除**：`TemplateConfig.agent_template` 无消费方，删除（含 `parser.py` 字段列表项） |
| `multi_turn_scheduler` / `gym_env` / `eval_metrics` | dev 已无适配器残留；`legacy_coverage.py` 的 `RLHF_REMOVED_ROLLOUT_FIELDS`（gym_env/use_gym_env/multi_turn_scheduler）与 `TRAIN_UNSUPPORTED_FIELDS`（batch_eval_metrics/include_for_metrics）拒绝表**保留**，让 legacy 用户拿到明确的"已移除 + 替代"报错 |

# 六、CLI flag 改名与 legacy 别名

Config 字段改名（均带 `#:` 注释）：
- `RLHFConfig`：`reward_funcs`→`orm`、`reward_weights`→`orm_weights`，新增 `prm`/`prm_weights`，删 `reward_model_plugin`。
- `InferConfig`：`infer_backend`→`sampler`，删 `prm_funcs`/`orm_model`/`prm_model`（模型 id 现在直接作为 `--orm`/`--prm` 的一项传入）。
- `TrainConfig`：`loss_type`→`loss`、`lr_scheduler_type`→`lr_scheduler`，删 `callbacks`；`optimizer` 保留为真优化器选择（`Literal['adam','sgd','muon','dist_muon']`）。
- `TunerConfig`：`tuner_type`→`tuner`。
- `TemplateConfig`：删 `agent_template`；`PluginConfig`：删 `custom_register_path`（统一到 `--external_plugins`）。

`legacy_coverage.py::LEGACY_ALIASES` 收录 1:1 改名：`reward_funcs→orm`、`reward_weights→orm_weights`、`prm_funcs→prm`、`loss_type→loss`（owner-aware，仅监督损失命令生效）、`lr_scheduler_type→lr_scheduler`、`tuner_type→tuner`、`infer_backend→sampler`、`sampler_engine→sampler`。`removed_option`（各带专属 replacement）收录二合一/删除项：`orm_model`、`prm_model`、`reward_model_plugin`、`custom_register_path`、`agent_template`、`callbacks`。

> `Any`/`List[Any]` 字段（`--sampler`、`--orm`、`--prm`）无法被 `HfArgumentParser` 从字符串强转，`parser.py::_patch_type_hints` 把它们对 argparse 呈现为 `str`/`List[str]`（CLI 值本就是名字/id/路径/类名字符串，dataclass 字段仍保留 `Any` 供程序化构造传入类对象）。

# 七、给用户的迁移指南

**多数插件文件不用改。** 老写法继续有效（`ORM` 就是 `RewardPlugin`，`orms['my'] = MyORM` 与装饰器写同一个 dict）。推荐新写法多一层注册时校验：

```python
from swift.dev.plugin import PluginRegistry, RewardPlugin

@PluginRegistry.register('reward', 'my_reward')
class MyReward(RewardPlugin):
    def __call__(self, completions, **kwargs):
        return [1.0] * len(completions)
```

启用：`--external_plugins /path/to/my_plugin.py --orm my_reward`。`--external_plugins` 每项可为 本地文件 | 本地文件夹 | hub id。`self.args` 是本次运行的 Config。

**新增一整类扩展点**（不改 swift 源码）：

```python
from swift.dev.plugin import PluginRegistry, SwiftPlugin

class MyKindBase(SwiftPlugin): ...
KIND = PluginRegistry.register_kind('my_kind', MyKindBase, config_field='my_kind_impl')
```

> 新 kind 必须有 Config 字段 + 消费方，否则不变式测试会失败——"注册了但没人读"正是本机制要消灭的东西。

# 八、测试位置

| 文件 | 内容 |
|---|---|
| `swift/dev/tests/component/plugin/test_registry.py` | 机制行为：注册/解析/形状校验/同名文件不互相覆盖/幂等/错误消息 |
| `swift/dev/tests/component/plugin/test_invariants.py` | 不变式：kind 必须可选且被消费 / 插件字段不许被静默忽略 / 不给 twinkle 传名字字符串 |
| `swift/dev/tests/component/config/test_all_cli_coverage.py` | 每个命令的 legacy 字段分类穷尽（direct/alias/unsupported），owner-aware 别名 |
