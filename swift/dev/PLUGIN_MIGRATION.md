# 插件迁移结果表（legacy `--external_plugins` 各 map → `swift/dev/plugin.py`）

> 记录插件机制从 legacy 迁到 dev 的现状、结论与依据。与 `MODEL_MIGRATION.md`/`DATASET_MIGRATION.md` 同体例：**只记已经落地的**，未落地的写清卡在哪，不留悬空。

> 一句话结论：**扩展点由 swift 自己声明、自己给基类、自己解析名字**（`swift/dev/plugin.py`），twinkle 只收到构造好的对象；legacy 的 11 个模块级 dict 里，reward 一类已经接入且**与老字典是同一个对象**（老写法 `orms['x'] = cls` 与新装饰器写同一处），其余各类要么归到内核（loss），要么在 dev 里**确实没有消费方**——后者被 `swift/dev/tests/component/plugin/test_invariants.py` 逐条钉住，带原因，不允许悄悄忽略。

## 判定规则
- **已接入（wired）**：dev 有 `PluginKind` 声明 + Config 字段 + 消费方，且有行为测试。
- **归内核（kernel）**：这类"插件"在 dev 里不是扩展点，而是 twinkle 的内核构件（loss / optimizer / metric 由 twinkle 提供实现）。
- **无消费方（unwired）**：Config 字段还在（`--xxx` 能解析），但 dev 里没有任何代码读它，因此**传了不生效**。原因逐条登记在不变式测试的 `UNWIRED` 表里。
- **不是扩展点（n/a）**：dev 明确拒绝或本就不该由插件表达。

---

# 一、legacy 的 11 个 map 逐个结论

计数为实测（`len(dict)`，2026-09）。

| legacy 注册表 | 位置 | 条数 | dev 结论 | 依据 / 备注 |
|---|---|---|---|---|
| `orms` | `swift/rewards/orm.py` | 8 | **已接入** | `swift/dev/rewards/orm.py` 尾部 `REWARD = PluginRegistry.register_kind('reward', (RewardPlugin, AsyncRewardPlugin), config_field='reward_funcs', entries=orms)`；`REWARD.entries is orms` 为 True（有测试） |
| `prms` | `swift/rewards/orm.py` | 2 | **已接入**（同一 kind） | PRM 与 ORM 同签名，`SamplingConfig.prm_funcs` 走同一个 `get_reward_funcs`；不另开扩展点 |
| `rm_plugins` | `swift/rewards/rm_plugin.py` | 2 | **无消费方** | dev PPO 用 `_build_reward_models` 直接建奖励**模型**，legacy 的"每个 RM 一个 plugin"钩子在 dev 没有对应位置；`RLHFConfig.reward_model_plugin` 传了不生效 |
| `loss_map` | `swift/loss/` | 7 | **归内核** | dev 的 loss 由 twinkle 提供（`swift.dev.loss` 只 re-export，`naming.py::resolve_loss` 查 twinkle 的 `torch_loss_mapping`）。若要开成扩展点，唯一允许 `base` 指向 twinkle 的就是这一类（`PluginKind.base` 已支持） |
| `loss_scale_map` | `swift/loss_scale/` | 7 | **已接入（间接）** | `TemplateConfig.loss_scale` 由 `builders/template.py` 传给 legacy `get_template`，由 legacy 自己的 loss_scale 注册表解析——dev 不重复造一个。注意 `RLHFConfig.loss_scale` 是**另一个字段且无消费方** |
| `callbacks_map` | `swift/callbacks/` | 7 | **无消费方** | legacy callback 是 HfTrainer 的 `TrainerCallback`；dev 的循环是 twinkle 的，没有可挂载对象。开 dev callback 点属于新设计 |
| `eval_metrics_map` | `swift/metrics/` | 5 | **无消费方** | metric 在 twinkle 是加到 optimizer status 的 `Metric` 对象；`TrainConfig.eval_metric` 这个名字 dev 从不读 |
| `optimizers_map` | `swift/optimizers/` | 6 | **不是扩展点（显式拒绝）** | `swift/dev/cli/sft.py:65-68` 对 `--optimizer` 抛 `NotImplementedError`（legacy 把它派发给基于 `create_optimizer` 的回调，dev 无等价物）。注意 `TrainConfig.optimizer` 是**另一回事**：`Literal['adam','sgd','muon','dist_muon']`，真优化器选择，`config/validate.py` 有消费方 |
| `tuners_map` | `swift/tuner_plugin/` | 3 | **不是扩展点** | tuner 是能力不是钩子：dev 走 `swift/dev/adapter.py::apply_tuner` + `TunerConfig`，加一种 tuner 是加实现，不是注册一个名字 |
| `agent_template_map` | `swift/agent_template/` | 32 | **无消费方** | agent template 在 legacy 是模板内部的工具调用格式化；dev 模板继承 legacy 行为但不暴露选择器，`TemplateConfig.agent_template` 传了不生效 |
| `multi_turns` | `swift/rollout/multi_turn.py:961` | 4 | **无消费方** | dev GRPO 没有多轮 rollout，scheduler 没有循环可调度；`RLHFConfig.multi_turn_scheduler` 传了不生效 |
| `envs` | `swift/rollout/gym_env.py:127` | 1 | **无消费方** | 同上，依赖多轮 rollout |

> 表里 13 行、legacy 侧 11 个 dict（`orms`/`prms`/`rm_plugins` 同住一个模块）。

## 为什么"无消费方"要留字段而不是删

删字段会让 legacy 的命令行在 dev 上直接报未知参数，迁移期不可接受；留字段又必须避免"参数被吃掉"。折中是：**字段留着，但每一条都在不变式测试里带原因登记**，并且测试会在它某天被接上时反过来报错（要求把它从 `UNWIRED` 移走）。

---

# 二、机制差异

| 维度 | legacy | dev |
|---|---|---|
| 注册表 | 11 个模块级 dict 字面量 | `PluginRegistry.KINDS`（kind 本身可注册）+ 每个 kind 的 `entries` |
| 注册方式 | `orms['my'] = MyORM`（改 dict） | `@PluginRegistry.register('reward', 'my')`，**或**照旧改 dict——两者是同一个 dict |
| 形状校验 | 无（跑到一半才炸） | 注册时 `issubclass` 检查，错的类型当场拒绝 |
| 基类 | `ORM` / `AsyncORM` 等各类各一 | `SwiftPlugin` 族；`ORM = RewardPlugin`、`AsyncORM = AsyncRewardPlugin` 是**别名**，老插件文件不用改 |
| 加载 | `BaseArguments._import_external_plugins`（`base_args.py:142`）import 副作用 | `PluginRegistry.load_configured(model_config)`，唯一入口在 `TrainAssembly.prepare()` |
| 模块名 | `importlib` 按文件名 | `swift_dev_plugin_<sha1(abspath)[:8]}_<stem>`：两个都叫 `plugin.py` 的文件不会互相覆盖 |
| 一个 kind 多种契约 | 不表达 | `PluginKind.base` 支持 tuple（reward 同时收 sync/async） |

## 为什么不复用 twinkle 的 `Plugin.load_plugin` / `construct_class`

三条都实测过：

1. **签名/语义不同**：swift 的 reward 是 `(completions, **columns) -> List[float]`（legacy 每个 ORM、每个用户插件文件都这么写）；twinkle 的 `Reward` 打分 `Trajectory` 对象，且**在 twinkle 内部零消费方**（它服务于 twinkle 自己的 cookbook）。
2. **表达不了"名字 + 本地 .py"**：`Plugin.load_plugin` 只认 `hf://`/`ms://` id 并强制 `trust_remote_code`——CLI 唯一能提供的接口（一个名字 + 一个本地文件）在那里没法表达。
3. **固定模块名会串**：twinkle 把每个插件都 `importlib.import_module('__init__')`，第二个插件命中 `sys.modules` 缓存，拿到的是第一个的类。

所以分工是：**名字由 swift 解析，twinkle 只收对象或类**。这条被 `test_no_name_string_is_handed_to_twinkles_loader` 钉住——dev 里任何 `model.set_loss('name')` 这类写法都会让测试失败。

---

# 三、插入点唯一化

插件加载**只有一份实现，也只有一处训练侧调用点**：

```
TrainAssembly.prepare()            # swift/dev/recipe/assembly.py
├── PluginRegistry.load_configured(model_config)   # external_plugins + custom_register_path
└── validate_configs(...)                          # 校验在加载之后：插件可以注册"校验要检查的东西"
```

`run_sft` / `run_embedding` / `run_seq_cls` / `run_reranker` 通过 `TrainAssembly.fit()` 走到它；`run_dpo` / `run_gkd` / `run_grpo` / `run_ppo` 形状不同（参考模型 / 教师 / 采样器），显式调 `prepare()` 后自己驱动分阶段方法。两条非训练路径单独调 `load_configured`：

| 路径 | 为什么也要加载 |
|---|---|
| `run_sampling` | 它按名字解析 `reward_funcs`/`prm_funcs`，用户自定义 ORM 不 import 就是 "not registered" |
| `run_infer` | 自定义模型/数据集注册在插件文件里，必须早于第一次按名字查找 |

---

# 四、给用户的迁移指南

**多数插件文件不用改。** 老写法继续有效：

```python
# my_plugin.py —— legacy 写法，dev 直接可用
from swift.dev.rewards import ORM, orms      # ORM 就是 RewardPlugin

class MyReward(ORM):
    def __call__(self, completions, **kwargs):
        return [1.0 if 'yes' in c else 0.0 for c in completions]

orms['my_reward'] = MyReward                  # 与装饰器写同一个 dict
```

推荐的新写法（多一层注册时校验）：

```python
from swift.dev.plugin import PluginRegistry, RewardPlugin

@PluginRegistry.register('reward', 'my_reward')
class MyReward(RewardPlugin):
    def __call__(self, completions, **kwargs):
        return [1.0] * len(completions)
```

用 `--external_plugins /path/to/my_plugin.py --reward_funcs my_reward` 启用。`self.args` 是本次运行的 Config（`cosine_*` / `repetition_*` 这类超参从它读）。

**新增一整类扩展点**（不改 swift 源码）：

```python
from swift.dev.plugin import PluginRegistry, SwiftPlugin

class MyKindBase(SwiftPlugin): ...
KIND = PluginRegistry.register_kind('my_kind', MyKindBase, config_field='my_kind_impl')
```

> 注意：新 kind 必须有 Config 字段 + 消费方，否则 `test_every_extension_point_is_selectable_and_consumed` 会失败——这是故意的，"注册了但没人读"正是本文要消灭的东西。

---

# 五、已知 gap / 下一步

| # | 事项 | 现状 |
|---|---|---|
| 1 | legacy 与 dev 的 `orms` 仍是**两个对象** | 实测 `swift.rewards.orms is swift.dev.rewards.orms` → False（键相同）。dev CLI 吃 legacy `SftArguments`，`--external_plugins` 由 legacy 侧 import 一次，dev 侧再 import 一次——现在两侧都会注册，但注册进的是两个字典。彻底统一要等 dev 参数层不再借 legacy `BaseArguments` |
| 2 | loss 未开成扩展点 | 需要一个 `base` 指向 twinkle `Loss` 的 kind，机制已支持（`PluginKind.base` 允许 twinkle 基类），未做 |
| 3 | callback / metric / 多轮 rollout | 都属于"dev 侧还没有可挂载的位置"，登记在 `UNWIRED` 表 |

## 测试位置

| 文件 | 内容 |
|---|---|
| `swift/dev/tests/component/plugin/test_registry.py` | 机制行为：注册/解析/形状校验/同名文件不互相覆盖/幂等/错误消息（17 条） |
| `swift/dev/tests/component/plugin/test_invariants.py` | 三条不变式：kind 必须可选且被消费 / 插件字段不许被静默忽略 / 不给 twinkle 传名字字符串（另含两张表的防腐检查） |
