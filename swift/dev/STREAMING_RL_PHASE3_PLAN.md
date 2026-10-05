# 流式 RL —— async 指标 + version-span 子 plan（Phase 3）

> 本文件是 Phase 3 的**具体施工子 plan**（用户无法交互审批 plan，故落盘为 MD）。
> 权威交接见同目录 `STREAMING_RL_HANDOFF.md`（§4 version-span 架构定案、§9 Phase 3 待办）。
> 动手前必读 skill：`dev-module-authoring`（分层/落位/下沉/剃刀/fail-loudly/AST-only）、`llm-terminology`（中文术语）、写测试时 `testcase-planning`。
> 术语：staleness / publish / abort / resume / partial rollout / version-span / pin / mixin / MRO / window / backpressure / group / straggler / drain / episode / idle ratio 保留英文。
> 纪律：作者阶段 AST-only 验证（+ 真实 import + ruff + ephemeral harness 用完即删）；正式端到端测试延到用户明确要求，届时走 testcase-planning。本 MD 属本地文档，不入 git。

---

## 0. 目标与三大原则的落点

**目标**：给 per-sample 流式 driver 补上 async 运行期可观测性——(a) driver 侧计时器/计数器（staleness、drop、train/idle 时间），(b) 每条 sample 的 version-span（partial rollout 跨 publish 的版本跨度），(c) 经既有 tracker 逐 optimizer step 发射这些指标。

**三原则的具体落点**（用户核心要求：能下沉的下沉、swift 侧简洁易懂易用、对标 verl、可比 swift legacy）：

| 原则 | Phase 3 的兑现 |
|---|---|
| **能下沉的下沉 twinkle** | 计时器（train/idle/total）与计数器是**算法无关**的 driver 生命周期量 → 沉进 `DriverStats`，在 `StreamingDriver.run()` 里累加。GRPO/PPO/RFT/GKD 任何消费者都免费获得，零 dev 分支。 |
| **不能下沉的 swift 侧要简洁** | version-span 是**控制面**概念（引擎必须版本无关，见 §4 交接），填充点只能在 dev 的 collect 侧；发射复用**既有** `_extra_step_metrics` hook + `RunTracker.log` 的通用标量发射——**tracking.py 零改动**（见 §3.4，修正 §9 的假设）。 |
| **对标 verl** | verl 的 async 指标核心是 trainer/rollouter 的 idle(bubble) ratio、staleness/off-policy 比例、partial rollout 跨度。本 plan 逐条对齐**可在 driver seam 观测的**那部分；不可观测的（rollouter GPU idle）**诚实标为非目标**并给理由，不伪造。 |
| **可比 swift legacy** | 这些是**新增**可观测性，不改任何训练数值/loss/step 语义；legacy 固定批路径（`_run_sync`/`_run_async`）不产生这些指标（无 driver），指标只在流式路径出现，回归面为零。 |

---

## 1. 施工清单（复制此清单跟踪）

```
Phase 3 施工进度：
- [x] 1. twinkle DriverStats 加计时器字段（train_active_time / idle_time / total_time）
- [x] 2. twinkle StreamingDriver.run() 三处计时包裹（consume / sleep / 全程）
- [x] 3. dev RolloutSample 加 version_span 字段
- [x] 4. dev _collect_per_sample 填 version_span（strategy-aware）
- [x] 5. dev _consume_streaming 暂存本 batch 的 version-span 列表
- [x] 6. dev StreamingLoopMixin 覆写 _extra_step_metrics（async 指标折叠 + super() 组合）
- [x] 6b. dev PPOLoop 补 _extra_step_metrics 接缝 + _record_step 调用它（冷审新增，见 §1 触及文件 4）
- [x] 7. AST（4 文件）+ 真实 import + ruff
- [x] 8. review-mode 冷审（不变量 + 反例 + 分层/复用/剃刀审计）
- [x] 9. ephemeral harness（计时器累加 + version-span 双策略 + 指标折叠组合，每条配 control）→ 用完即删
- [ ] 10. 向用户汇报 Phase 3 完成（结论先行、简练、禁内部编号），等签收
--- 正式端到端测试是用户触发的独立阶段，非本 Phase 完成门禁 ---
```

**触及文件（精确 4 个）**：
1. `twinkle/src/twinkle_agentic/async_rl/streaming_driver.py` —— 步骤 1-2（下沉）
2. `swift/dev/rollout/__init__.py` —— 步骤 3（version_span 字段）
3. `swift/dev/recipe/_streaming_loop.py` —— 步骤 4-6（dev 接线）
4. `swift/dev/recipe/run_ppo.py` —— 步骤 6b（给独立的 `PPOLoop` 补 `_extra_step_metrics` 接缝）。
   **冷审新增**：原 §3.4 假设「GRPO/PPO 都经 TrainLoop 的 `_extra_step_metrics` hook」被证伪——
   `PPOLoop` 是独立类（`class PPOLoop:` 直接继承 object，非 TrainLoop 子类），其 `_record_step(mean_reward)`
   内联建 record、**不经任何 hook**，故 mixin 覆写对 PPO 是死代码（async 指标永不发射，且 `super()` 会
   AttributeError）。修法：给 `PPOLoop` 补一份与 TrainLoop 同名的 `_extra_step_metrics`（默认返回 `{}`）
   并在其 `_record_step` 里 `record.update(self._extra_step_metrics(metrics))`，mixin 的 `super()` 即命中它。
   这是「复用既有接缝命名」而非另造新钩子——PPO 由此获得与 TrainLoop 一致的扩展点。

**`tracking.py` / `grpo.py` / `ppo_async.py` 仍零改动**（§3.4 论证成立：`RunTracker.log` 通用标量发射，
GRPO 经 TrainLoop 的 `_record_step`→`_extra_step_metrics` 已命中 mixin 覆写）。

---

## 2. 下沉部分：twinkle DriverStats 计时器（步骤 1-2）

### 2.1 设计决策（解决 §9 开放项「计时器怎么暴露给 dev」）

**决策：走既有 `self.stats`（`DriverStats`），不新增 seam、不改 `run()` 返回值。**

理由：`DriverStats` 已经是 driver 的对外计数契约（`admitted/collected/consumed/off_policy_consumed/dropped_stale/...`），dev 侧 `self._streaming_driver.stats` 在 `run()` 期间即已可读（`_streaming_loop.py:253` 先赋值 driver，`254` 才 `run()`；consume 发生在 run 内）。计时器与计数器同源同生命周期，加字段是最小改动、零新接缝——符合剃刀。新增 seam 或改返回值都会扩大接触面且无收益。

### 2.2 具体改动

`DriverStats`（`streaming_driver.py`，现有一组 int 计数器后）新增 3 个 float 字段，每个带 `#:` 注释：

```python
train_active_time: float = 0.0   # 阻塞在 consume()（训练）上的墙钟秒数（driver 线程）
idle_time: float = 0.0           # 阻塞在 sleep 分支（有 in-flight、无 completion、buffer 不可训）的墙钟秒数
total_time: float = 0.0          # 整个 run() 的墙钟秒数（在 finally 里定）
```

**只加这 3 个，不加 `gen_active_time`。** §9 原列 `gen_active_time`，但 driver 线程不生成——生成在 disaggregated sampler 后台跑，driver 只 poll。`total − train − idle` 是 driver 的准入/poll/collect/publish 开销残差（不是 sampler GPU 时间）。要真正测生成 GPU 时间必须插桩 sampler（数据面），会反转控制面/数据面分层——**非目标**（§5 记录）。dev 侧要「生成占比」时用 `1 − train_ratio − idle_ratio` 得到 driver-overhead 占比即可，语义诚实。

`run()` 三处包裹（`time.perf_counter()`，`time` 已 import）：

- **train**（现 `streaming_driver.py:380` 的 `self._train_pull(records, ready)`）：
  ```python
  _t = time.perf_counter()
  self._train_pull(records, ready)
  self.stats.train_active_time += time.perf_counter() - _t
  ```
- **idle**（现 `393` 的 `self._sleep(0.01)`）：
  ```python
  _t = time.perf_counter()
  self._sleep(0.01)
  self.stats.idle_time += time.perf_counter() - _t
  ```
- **total**：`run()` 体开头（`register_context` 后）记 `_run_start = time.perf_counter()`；`finally` 末尾（现 `407` `final_version` 赋值处）`self.stats.total_time = time.perf_counter() - _run_start`。

**不变量**：`total_time >= train_active_time + idle_time`（残差 = driver overhead ≥ 0）；三值单调不减；drain/budget/error 三条退出路径都在 finally 里定 total_time（不遗漏）。

---

## 3. dev 侧：version-span + 指标发射（步骤 3-6）

### 3.1 RolloutSample.version_span（步骤 3）

`swift/dev/rollout/__init__.py` 的 `RolloutSample` dataclass，在 `policy_version: int = 0` 后追加：

```python
#: partial rollout 跨 publish 的版本跨度：这条 trajectory 从准入到 collect，策略版本推进了几次。
#: adapter_snapshot 下恒为 0（整条 episode pin 在一个 frozen adapter_path，逐轮同版本）；
#: in_place 下 = collect 时 current_version − 准入版本（episode 跨版本，被 abort+resume 到新权重上）。
#: 仅指标用途，无正确性消费者（staleness 锚点始终是最老的准入版本 policy_version）。
version_span: int = 0
```

有默认值、追加在末尾 → 不破坏既有构造/`_finalize_samples`/字段序。

### 3.2 version_span 填充点（步骤 4）

`swift/dev/recipe/_streaming_loop.py` 的 `_collect_per_sample`（现 293-294）。`self.rollout.collect_sample(handle)` 已 stamp `sample.policy_version = handle.policy_version`（准入版本），故 span 从 `sample.policy_version` 起算，与 driver collect 处（`streaming_driver.py:344` 的 `current`）同一时刻读同一 `current`，一致：

```python
sample = self.rollout.collect_sample(handle)
if self._weight_sync_strategy == 'in_place':
    current = self._ctx_mgr.get_rollout_policy(self._ctx).version
    sample.version_span = max(0, current - sample.policy_version)
# adapter_snapshot：保持默认 0（整条 episode pin 一个 frozen path，逐轮同版本）
return self._finalize_samples([sample], [handle.prompt_idx])[0]
```

`_ctx_mgr`/`_ctx`/`_weight_sync_strategy` 都是 `_init_streaming` 已建的 mixin 属性；`max(0, …)` 防负（版本只增，理论不减，但 fail-safe 不为负）。**引擎（twinkle MultiTurnRollout / sampler）绝不感知 version**——填充只在 dev 控制面侧。

### 3.3 本 batch version-span 暂存（步骤 5）

`_consume_streaming`（现 366-376）在把 records 交给 `_consume_async_samples` 前暂存本 batch 的 span 列表，供步骤 6 的逐 step 发射读取（一次 consume 跑 `ga` 个 optimizer step，这些 step 共享同一 batch 聚合——batch 级信号，语义正确）：

```python
def _consume_streaming(self, records) -> int:
    samples = [record.sample for record in records]
    self._last_batch_version_spans = [int(getattr(s, 'version_span', 0)) for s in samples]
    self._consume_async_samples(samples)
    return self._streaming_step_delta()
```

`_last_batch_version_spans` 在 `_init_streaming` 里初始化为 `[]`（避免首个 step 前无属性）。

### 3.4 指标发射：覆写 `_extra_step_metrics`（步骤 6）—— tracking.py 零改动

**关键论证（修正 §9 假设）**：`train_loop.py:424` 的 `record.update(self._extra_step_metrics(metrics))` 已是逐 optimizer step 的通用扩展点，`RunTracker.log`（`tracking.py`）把 record 里**任意** int/float 非 `step` 标量原样发射到 tensorboard/wandb/swanlab。所以 async 指标只是 record 里的额外键——**`tracking.py` 不需要任何新字段/新逻辑**，`grpo.py`/`ppo_async.py` 也不改。这是剃刀式的最小接线。

`StreamingLoopMixin` 覆写 `_extra_step_metrics`，**先 `super()` 再加 async 指标**（MRO：`StreamingGRPOLoop(StreamingLoopMixin, GRPOLoop)`，mixin 的 `super()` 命中 `GRPOLoop._extra_step_metrics` 的 entropy/rollout_log_ratio；PPO 同理命中其自身版本——组合不丢字段）：

```python
def _extra_step_metrics(self, metrics: dict) -> dict:
    extra = super()._extra_step_metrics(metrics)
    driver = getattr(self, '_streaming_driver', None)
    if driver is None:
        return extra                     # 非流式路径（防御：不会发生，但稳固）
    stats = driver.stats
    # 时间占比取「自上一个 optimizer step 的 fold 以来」的 delta——对齐 verl 的 per-step ratio 语义
    # （累计比会被长跑平滑掉）。注意 _extra_step_metrics 每个 optimizer step 都被调用（record 每步都建，
    # tracker.log 只按 cadence 决定写不写），故 delta 是 per-step、非 per-log-interval。
    d_train = stats.train_active_time - self._stats_snapshot['train']
    d_idle = stats.idle_time - self._stats_snapshot['idle']
    self._stats_snapshot = {'train': stats.train_active_time, 'idle': stats.idle_time}
    denom = d_train + d_idle   # 选项 A 的分母（见下）；run 中 total_time 恒 0，不能用它
    spans = self._last_batch_version_spans or [0]
    extra.update({
        'trainer_idle_ratio': (d_idle / denom) if denom > 0 else 0.0,
        'partial_ratio': sum(1 for s in spans if s > 0) / len(spans),
        'max_partial_span': max(spans),
        'version_span_mean': sum(spans) / len(spans),
        'off_policy_consumed': stats.off_policy_consumed,   # 累计（罕见事件，看上升曲线）
        'dropped_stale': stats.dropped_stale,               # 累计
        'stream_publishes': stats.publishes,                # 累计
    })
    return extra
```

> **剃刀修正（冷审后）**：原设计还发 `train_time_ratio = d_train/denom`，但选项 A 的分母就是
> `d_train + d_idle`，故 `train_time_ratio ≡ 1 − trainer_idle_ratio`，是冗余标量——**已删**，只留可操作的
> bubble 比 `trainer_idle_ratio`。

**run 中 total_time=0 的处理（重要设计点）**：`total_time` 只在 finally 里定，run 进行中恒为 0。所以逐 step 发射时**不能**用 `stats.total_time` 做分母——改用 delta 分母 `d_total = d_train + d_idle + d_overhead`。但 driver overhead（准入/poll/collect/publish）在 run 中无独立累加器。两个选项：

- **选项 A（推荐，最简）**：分母直接用 `d_train + d_idle`，即「训练 vs 空转」的二分占比（忽略 driver overhead，它通常远小于二者）。指标名诚实：`trainer_idle_ratio = d_idle/(d_train+d_idle)` = 训练循环里 trainer 饿死（无就绪 batch）的时间占比。**这正是 verl trainer bubble ratio 的语义**。
- 选项 B：DriverStats 再加一个 `overhead_time` 累加器（准入+poll+collect+publish 段包裹）。多一处计时、多一个字段，收益仅是把 driver 自身开销从分母里扣除——driver overhead 相对 GPU 训练/生成极小，不值得。**否决（剃刀）**。

采纳**选项 A**：分母 `d_train + d_idle`，`_stats_snapshot` 只需存 `{'train','idle'}` 两个累计基线。

`_stats_snapshot` 在 `_init_streaming` 初始化 `{'train': 0.0, 'idle': 0.0}`。

**指标 → 来源对照表**（对标 verl）：

| dev 指标 | 来源 | verl 对应 | 观测性 |
|---|---|---|---|
| `trainer_idle_ratio` | driver delta：idle/(train+idle) | trainer bubble/idle ratio | ✅ driver 可测 |
| `partial_ratio` | 本 batch version_span>0 占比 | partial rollout 触发率 | ✅ collect 侧 |
| `max_partial_span` / `version_span_mean` | 本 batch span 极值/均值 | version-span 分布 | ✅ |
| `off_policy_consumed` | 累计 stats | stale/off-policy 样本数 | ✅ 已有计数 |
| `dropped_stale` | 累计 stats | staleness 丢弃数 | ✅ 已有计数 |
| `stream_publishes` | 累计 stats | weight sync 次数 | ✅ 已有计数 |
| ~~`rollouter_idle_ratio`~~ | —— | rollouter bubble ratio | ❌ **非目标**（§5） |
| ~~`gen_active_time`~~ | —— | 生成 GPU 时间 | ❌ **非目标**（§5） |

---

## 4. review-mode 冷审要点（步骤 8，动手后执行）

不变量逐条推反例并实跑：
- `total_time >= train + idle`（含 drain/budget/error 三退出路径都在 finally 定 total）——反例：consume 抛错时 total 是否仍定？（finally 保证）
- version_span 恒 ≥ 0（`max(0, …)`）；adapter_snapshot 恒 0；in_place 下跨一次 publish 的 episode span==1——反例：admit 与 collect 之间发生 2 次 publish，span 应 ==2。
- `_extra_step_metrics` 的 `super()` 组合**不丢** GRPO 的 entropy/rollout_log_ratio——反例：断言合并后两类键都在。
- delta 分母为 0（首个 step 前、或该 log 区间无 train 无 idle）时 ratio 不除零——反例：d_train+d_idle==0 → ratio 取 0.0。
- `getattr(self,'_streaming_driver',None)` 防御分支：非流式 loop 调 `_extra_step_metrics` 不炸。
- MRO：mixin 覆写不遮蔽 GRPOLoop 版本（`StreamingGRPOLoop.__mro__` 里 mixin 在前、super() 链完整）。
- 剃刀审计：无死字段（version_span 有发射消费者）、无半成品、无「以后可能用」参数。
- 分层审计：引擎（twinkle MultiTurnRollout/sampler）零改动、零 version 感知；计时器/计数器在 twinkle（算法无关，正确下沉）；version-span 填充与指标折叠在 dev（控制面，正确）。

---

## 5. 非目标（诚实记录，不伪造指标）

- **`rollouter_idle_ratio`（sampler GPU 空转率）不实现**：dev 流式 driver 是**单线程控制循环**，生成委托给后台 disaggregated sampler；driver 观测不到 sampler 的 GPU idle。verl 能测是因为它直接插桩 Ray rollout worker。要拿到这个量必须插桩数据面（sampler），反转控制面/数据面分层、爆炸半径覆盖所有 sampler 消费者——违反本任务分层铁律。若将来确需，属**独立的 sampler 侧 telemetry 议题**，不在 driver 层。
- **`gen_active_time`（生成 GPU 时间）不实现**：同上。driver 侧只有 `total − train − idle` 的 driver-overhead 残差，非 sampler GPU 时间，不冒名。

---

## 6. 验证计划（步骤 7-9）

- **AST**：`ast.parse` 触及的 3 个源文件。
- **真实 import**：`twinkle_agentic.async_rl.streaming_driver`、`swift.dev.rollout`、`swift.dev.recipe._streaming_loop`、`swift.dev.recipe.grpo_async`（`/usr/local/bin/python`，`--import-mode=importlib` 若用 `-m`）。
- **ruff**：项目 select `B,C,E,F,W,I`（ignore F401/F403/F405/F821/E251，line-length 120，C901 不 gate）；零新增违规。
- **ephemeral harness（`/tmp`，用完即删，每条配 control 防空跑假绿）**：
  1. **计时器累加**：真实 `StreamingDriver` + fake seams + 注入 `sleep`，跑一个会触发 consume 与 sleep 分支的小 prompt 流；断言 `train_active_time>0`、`idle_time>0`、`total_time>=train+idle`。control：把 train 包裹去掉 → train 恒 0（红）。
  2. **version_span（in_place）**：fake RolloutEngine，admit 与 collect 间人为推进 current_version 2 次；断言 `sample.version_span==2`。control：填充逻辑改成恒 0 → 红。
  3. **version_span（adapter_snapshot）**：同场景但 strategy=adapter_snapshot；断言 span==0。
  4. **指标折叠组合**：构造 StreamingGRPOLoop 的 `_extra_step_metrics`（或用最小 fake loop 挂 mixin），喂含 loss_entropy 的 metrics；断言输出**同时**含 `entropy`（super 来的）与 `trainer_idle_ratio`/`partial_ratio`（mixin 加的）。control：去掉 super() 调用 → entropy 丢失（红）。
  5. **除零防御**：d_train+d_idle==0 时 `trainer_idle_ratio==0.0` 不抛。
  - harness 用 fake 组件即可（无 GPU，纯控制面）；正式端到端测试（真实 MultiTurnRollout + 脚本化 sampler）延到用户要求时走 testcase-planning。

---

## 7. Definition of Done

**Feature DoD（本 Phase 完成线，测试非门禁）**：
- [x] 本子 plan 落盘（本文件）。
- [x] 计时器/计数器下沉 twinkle `DriverStats`，算法无关、零 dev 分支。
- [x] version_span 字段 + strategy-aware 填充在 dev collect 侧，引擎零感知。
- [x] async 指标经既有 `_extra_step_metrics`+`RunTracker.log` 发射，tracking.py 零改动。
- [x] 冷审发现的 PPO 独立类缺陷已修（`PPOLoop` 补 `_extra_step_metrics` 接缝，run_ppo.py）。
- [x] `rollouter_idle_ratio`/`gen_active_time` 明确标非目标并给分层理由（不伪造）；冗余 `train_time_ratio` 已删（剃刀）。
- [x] 4 文件 AST + 真实 import + ruff 零新增违规（3 tracked 文件 ruff 与 HEAD 平价；streaming_driver 新增行无违规）。
- [x] review-mode 冷审执行，反例实跑（单轮/多轮 policy_version stamping 核实、PPO/GRPO MRO super() 命中核实）。
- [x] ephemeral harness 全绿（23/23，含 3 teeth control），用完即删。
- [ ] 向用户汇报 Phase 3 完成并等签收，再进 Phase 4。

**Test DoD**：仅在用户明确要求补测试时适用，届时走 testcase-planning（完整 + 端到端 + 反向验证），非本 Phase 门禁。
