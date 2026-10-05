# Phase 4 子 plan（Option A：sync 独立保留，只迁蒸馏 overlap + 退役 async-only 死代码）

> Phase 4 的**书面子 plan**（message 8：每 phase 动手前先写 MD 列出具体要做的事）。本地保留，**不提交**。
> 三原则落点：**正确性**（蒸馏 async 迁移后 consume 逐字复用现有 teacher-score+train，feature stack 不分叉；sync 路径行为不变）、**稳固性**（退役共享函数前逐个列全消费者、fail-loudly、删前重跑 grep）、**简洁性**（删 async-only 死代码不留半态，sync 路径不动）。

---

## 裁定依据（Option A：为什么不再走「sync=staleness=0 单旋钮」）

- **用户 Q3 逐字要求** = 「per sample 的 sync 路径不是不要，async 和 sync 是需要**可以选择的**」，**不是**「统一成一个旋钮」。selectable ≠ 同一 driver／同一旋钮——只要 `async_mode` 能选到 sync、也能选到 async 就达标。
- **verl 对标**（file:line 坐实）：verl 用 `trainer.v1.trainer_mode` 注册**三个各自独立的 trainer**——`sync`（`trainer_sync.py:24-42`，docstring「Trainer and rollout are **colocated**」，每步 `update_weights()`→`sleep_replicas()`→generate→train 的**简单阻塞循环**，无 staleness 概念、不走 async 驱动；v0 更直接 `ray_trainer.py:333 assert hybrid_engine`）、`colocate_async`、`separate_async`（`trainer_separate_async.py:65 assert rollout.nnodes>0`）。统一只发生在 **checkpoint-engine primitive 层**（`checkpoint_engine/base.py:467 sleep_replicas`/`472 wake_up_replicas`/`510 update_weights`），**不在训练 loop 层**；sync 没有 disaggregated 路径。**verl 没把 sync 折叠进 async 驱动。**
- **技术坐实（我们这边）**：StreamingDriver 按设计 **disaggregated-only**——`_collect_per_sample`（`_streaming_loop.py:339`）明写「a disaggregated sampler (**mandatory here**) never performed [colocate hand-over]」，故意不调 `finish_generate`；`run()`（`streaming_driver.py:299-301`）**第一遍不 publish**（`steps_in_cycle=0<K`），colocate 下首遍 admission 会在「trainer 占卡、sampler 未唤醒」时提交生成 → 崩。强迁 colocate sync = 回归常见廉价配置 + 在**默认路径**引入内存敏感的逐 pass GPU hand-over（正是 verl 用「sync 独立成类」绕开的事）。
- **结论**：**sync 全部保留 base 路径**（colocate+disaggregated 都支持）；**流式驱动只服务 overlap regime**（staleness≥1，validate 已强制 disaggregated）。「sync=staleness=0 单旋钮」是我此前的架构偏好、非用户需求，本 Phase 放弃。

---

## 0. 一句话目标

Phase 1-3 已把 GRPO/PPO 的 overlap regime（one_step_off / fully_async）迁到 per-sample `StreamingDriver`。Phase 4 收口剩下两件事：

1. **蒸馏族 `lmbda==1.0` 的 async（one_step_off）迁到 `StreamingGKDLoop`@staleness=1**（per-sample 流式，对齐 GRPO/PPO overlap；OPSD/MOPD 经继承自动获得，只覆盖 teacher 信号）。
2. **退役 async-only 死代码**（迁移后全消费者归零者）。

**不动**：所有 sync（GRPO/PPO/RFT/蒸馏的 `async_mode='none'`）保留在 base `_run_sync`（colocate+disaggregated 都支持）；RFT sync-only，`RFTLoop.fit` 原样保留；`_generate`/`finish_generate`/`sync_weights`/`PromptBatchScheduler`/`_train_rollout_batch`/`_consume_async_samples`/`_finalize_samples` 全部保留（sync 用）。

**退役集**（迁移后归零即删）：`overlap_rollout_batches`；GRPO/PPO/GKD base 的 `_run_async`+`_submit_generation`+`_collect_generation`（GRPO/PPO 的已死，GKD 的迁移后死）；batch-async 的 `submit_generate`/`collect_generate`/`cancel_generate`/`GenerationHandle`（含 submit_generate 的多轮 guard）。

---

## 1. 概念对齐

`StreamingDriver`（twinkle，算法无关）的控制循环：拿一个 prompt stream，把每个 prompt 的 `num_generations` 条轨迹**一条一条准入**（per-sample submit，每条 pin 一个 policy version），非阻塞 poll **先完成先收**进 ready buffer，`assembly_ready` 判定「凑够一个训练单元」就 `consume` 训练（GRPO 拉**完整 group**、PPO/蒸馏拉单样本），每 K 步 `publish` 一次权重。overlap regime 只是 staleness 旋钮（1=一步重叠、>1=深缓冲）；**sync 不在此旋钮上，走 base 阻塞路径**。

sampler 的 **continuous batching 正是让这套「逐条准入、完成即收、不等整批最慢那条」高效的原因**——这是流式相对老固定批（整批阻塞、有 straggler 尾巴）在 overlap regime 下的核心优势。

---

## 2. 现状事实（已回读源码坐实）

### 2.1 路由与类层次
- 类层次：`MOPDLoop→OPSDLoop→GKDLoop→GRPOLoop→TrainLoop`；`PPOLoop` 独立；`RFTLoop(GRPOLoop)` 自带拒绝采样 `fit`（run_rft.py:276）。
- **GRPO 路由**（run_grpo.py:503）：`async_mode in ('one_step_off','fully_async')` → `StreamingGRPOLoop`；`'none'` → base `GRPOLoop`（`async_generate=False`，run_grpo.py:574）。→ base `GRPOLoop._run_async` **已死**。
- **PPO 路由**（run_ppo.py:211）：同上（`async_generate=False`，run_ppo.py:249）。→ base `PPOLoop._run_async` **已死**。
- **GKD 路由**（run_gkd.py:203）：`async_generate=(async_mode=='one_step_off')`；`fit`（run_gkd.py:468）**直接** branch `async_generate` → `_run_async`（450，用 `overlap_rollout_batches`，**活**）/ `_run_sync`（434）。→ 本 Phase 唯一迁移对象。

### 2.2 迁移后各符号生死（Option A 修订）
| 符号 | 现消费者 | 迁移后 |
|---|---|---|
| GRPO/PPO base `_run_async`/`_submit_generation`/`_collect_generation` | base `_drive` async 分支（`async_generate` 恒 False，**已死**） | **死** → 删 |
| GRPO/PPO base `_run_sync`/`_drive`(sync 分支)/`_rollout_step` | base `_drive`（sync，`'none'`） | **保留**（sync 路径） |
| `GKDLoop._run_async` | GKD fit（async，**活**） | 迁 StreamingGKDLoop 后**死** → 删 |
| `GKDLoop._run_sync` | GKD fit（sync） | **保留**（所有蒸馏 sync，含 `lmbda<1`） |
| `RFTLoop.fit`（拒绝采样） | run_rft 入口（sync-only） | **保留**（RFT 不迁） |
| `_generate`（阻塞固定批） | `_rollout_step`(sync,保留) + RFT.fit(保留) + GKD `_round_rows` on-policy(保留) | **保留** |
| `overlap_rollout_batches` | grpo(死)/run_ppo(死)/run_gkd(活) | GKD async 迁流式后**全死** → 删 |
| `submit_generate`/`collect_generate`/`cancel_generate`/`GenerationHandle` | 仅 GRPO/PPO/GKD 的 `_submit_generation`/`_collect_generation`/`_run_async` | 全死后 → 删（多轮 guard 随 submit_generate 删） |
| `_train_rollout_batch`/`_consume_async_samples`(GRPO)/`_finalize_samples`/`PromptBatchScheduler`/`finish_generate`/`sync_weights` | sync + 流式 consume 共用 | **保留** |

### 2.3 蒸馏 async 迁移的落点（已坐实）
- 蒸馏 async（one_step_off）validate **已强制** `vllm_mode='disaggregated'`（validate.py:330，对所有 overlap regime）→ 与 StreamingDriver 的 disaggregated-only 一致，**无 colocate 问题**（这正是 Option A 下迁移可行、而 sync 迁移不可行的分界）。
- `StreamingGKDLoop(StreamingLoopMixin, GKDLoop)`：assembly=`_assembly_ready_ppo`（蒸馏无 group 契约，per-sample）；`_consume_async_samples` 复用 GKD 现有 teacher-score+train（run_gkd.py:430）；`_streaming_unit_size`=`train_batch_size*ga`。OPSD/MOPD 经继承自动获得（只覆盖 teacher 信号）。
- `GKDLoop.fit` 需重构为经 `_drive`（镜像 `GRPOLoop.fit`）：base `GKDLoop._drive`=`_run_sync`（sync-only，async 已路由走）；`StreamingGKDLoop` 用 mixin `_drive` 覆盖（MRO：mixin 在前）。
- 路由改：`async_mode=='one_step_off'` 且 `lmbda==1.0` → `StreamingGKDLoop`(staleness=1)；否则（`'none'`，或 `lmbda<1`）→ base `GKDLoop`(`async_generate=False`)。fully_async 对蒸馏仍拒。
- validate M12 多轮拒斥（validate.py:346-352）：蒸馏 async 迁流式后（per-sample `submit_sample`→per-episode 多轮）放开 `max_turns`；蒸馏 sync（阻塞 `_generate` 单轮）仍拒。

### 2.4 复用边界：单 LoRA/全参走通用控制面，**不走 server-client 多 LoRA 架构**（已回读源码坐实）

dev 流式路径从 `twinkle_agentic.async_rl` **只**导入四样通用件（`_streaming_loop.py:64-67`）：`context_manager.RLContextManager`、`streaming_driver.StreamingDriver`、`types`、`weight_sync`。生成走 **dev 自己的 `RolloutEngine.submit_sample/poll_completions/collect_sample`**（线程内 one-episode `MultiTurnRollout.generate` + 核心 `sampler.sample()`），权重同步走 `rollout.sync_weights` → `checkpoint_engine.CheckpointEngineManager`（naive/colocate/standalone）。partial rollout 的 abort/resume 是 **核心 sampler 能力**（`twinkle/sampler/partial_rollout.py` 的 `PartialRolloutMixin`，mix 进普通 `vLLMSampler`/`SGLangSampler`）。

**明确排除、单 LoRA/全参不应使用**：twinkle 的 **server-client / TransferQueue 多 LoRA 多租户栈**——`async_rl/native_tq`、`async_rl/pipeline`（`AsyncMultiLoraGRPOPipeline`）、`async_rl/workers`、`async_rl/scheduler`、`async_rl/data_plane`、`async_rl/generation_submissions`（复数，配 `*_sampler_tq`）、server 的 `deployment`/`router`/`gateway`。这套是「多个 LoRA tenant 共享一份 base + 一个 sampler、经 HTTP server-client 队列交换 group」的多租户 serving 场景；单 LoRA/全参只有一个 policy、一个训练进程，无多租户要 serve，硬走只会白背 TransferQueue + Ray RPC + server 开销。dev 现路径**已不导入**这些（grep 坐实），Phase 4 迁移不得引入。

**易混点钉死**：核心 `twinkle/sampler/generation_submission.py`（单数，vllm/sglang sampler 用）≠ `twinkle_agentic/async_rl/generation_submissions.py`（复数，TransferQueue/多 LoRA 用）。dev 走前者所在的**核心 sampler**，不碰 `_tq` 后缀的多租户件。

**大集群扩容（对标 verl 的正确锚点）**：单策略扩容 = **sampler 引擎自身 data-parallel** + **`CheckpointEngineManager` mode='standalone' 经 NCCL/HCCL fan-out 到所有 sampler DP rank**（`build_topology` 用 `sampler.device_mesh.data_world_size`）。verl 式「多副本 + server router 负载均衡」在 twinkle 里对应多租户 serving 层，**我们故意不借**。

**已知局限（诚实记录，非 Phase 4 范围）**：verl 有 `delta_checkpoint_engine`（稀疏 diff，只传变化权重）；twinkle `CheckpointEngineManager` 目前只有全量或 LoRA 增量（`merge_and_sync=False`），**无 delta diff**。全参数 in_place 每步全量传输的大集群带宽成本高于 verl delta。属 twinkle 侧能力缺口，dev 接线补不了，列为后续 twinkle 下沉候选。

---

## 3. 关键设计决策（Option A 后）

### ✅ D3' — sync 全保留 base，只迁蒸馏 overlap
取代原 D1/D2/D3/D4。原 **D1**（staleness=0 强制 K=1）、**D2**（staleness=0 in_place guard 放松）、**D4**（RFT publish 节奏）**全部消失**——不再有 staleness=0 流式，RFT 不迁。据此：
- `_check_streaming_config`（_streaming_loop.py:107）的 `max_staleness < 1: raise` **保持原样**（流式仅 staleness≥1）。
- in_place 的 partial-rollout guard **保持原样**（staleness≥1 publish 点恒有 in-flight，需 abort+resume）。
- `lmbda<1` 混合蒸馏 sync 天然保留 base（off-policy round 不生成，本就无法流式化）；`lmbda==1.0` 蒸馏 sync 也保留 base（Option A：所有 sync 走 base）。

### ✅ D6 — 退役采「先确认归零后删」，迁移与删除分两个 checkpoint
删每个共享函数前重跑 grep 确认消费者归零（§4）；ckpt-1（蒸馏 async 迁移）先签收，ckpt-2（删死代码）再执行。避免默认路径回归时无法回退。

---

## 4. 退役共享函数的全量消费者枚举（已 grep 坐实，删前重跑复核）

**`overlap_rollout_batches`（train_loop.py:58）**：grpo.py:1024（`_run_async`，**死**）、run_ppo.py:681（**死**）、run_gkd.py:460（`GKDLoop._run_async`，**活**）。→ GKD async 迁流式 + GRPO/PPO `_run_async` 删除后三者全死 → 删。文档引用（rollout_config.py:89、validate.py:242/251/280/364）同步更新。

**`submit_generate`/`collect_generate`/`cancel_generate`/`GenerationHandle`（rollout/__init__.py:524/…/134）**：消费者仅 GRPO `_submit_generation`(301)/`_collect_generation`(319)、PPO 同名(609/619)、GKD 经继承用 GRPO 的（run_gkd.py:462-463）。→ 三条 async batch 路径全删后归零 → 整体删（**多轮 guard 随 submit_generate 删除**）。删前重跑 grep 复核无 twinkle 侧同名依赖、`SampleHandle` docstring 交叉引用已清。

**GRPO/PPO base `_run_async`/`_submit_generation`/`_collect_generation` + `GKDLoop._run_async`**：见 §2.2。**保留** `_run_sync`/`_drive`(sync 分支)/`_rollout_step`/`_generate`/`finish_generate`/`sync_weights`/`PromptBatchScheduler`/`_train_rollout_batch`/`_consume_async_samples`/`_finalize_samples`/`RFTLoop.fit`（sync 路径用）。

**旧测试**：直接实例化 base loop 或走 GKD `_run_async` 的测试删除后会红——作者阶段 AST-only 不跑，标记为 UT 阶段更新。

---

## 5. Review-mode 冷审计划（对实际代码执行）

1. **不变式**：
   - 蒸馏 async 迁移后 `_consume_async_samples`（GKD teacher-score+train）逐字复用，feature stack 不分叉；OPSD/MOPD 经继承拿到同一 consume。
   - `StreamingGKDLoop` 的 mixin `_drive` 覆盖 base `GKDLoop._drive`（MRO：mixin 在前）；base `GKDLoop` 仍 sync-only。
   - staleness=1 蒸馏：每 pull version 推进与 one_step_off 语义一致；assembly per-sample（蒸馏无 group 契约）。
   - 路由：`'none'`→base、`one_step_off`+`lmbda==1.0`→Streaming、`one_step_off`+`lmbda<1`→拒（validate 已有）、fully_async 蒸馏→拒。
   - **sync 路径行为逐字不变**（colocate+disaggregated 都仍可选、仍走 base `_run_sync`）。
2. **推反例并实跑（harness）**：
   - 蒸馏 `lmbda==1.0` one_step_off：Fake teacher + 脚本化 sampler，跑通 staleness=1，consume 的 sample 集合 == 迁移前 `overlap_rollout_batches` 路径产出（内容/顺序一致）。
   - `lmbda<1` 仍走 base sync（不被误路由到流式）；fully_async 蒸馏被拒。
   - OPSD/MOPD 经继承跑通（MRO 不破坏 teacher 信号覆盖）。
   - 退役后：grep 确认 `overlap_rollout_batches`/`submit_generate`/`collect_generate`/`cancel_generate`/`GenerationHandle` 零消费者；真实 import 全触及模块。
   - drain/短流（prompt 数<pull_rows）/reached_max：无 pin 泄漏、`list_live_partitions()==[]`。
3. **复用/落位/分层审计**：`StreamingGKDLoop` 复用 mixin（不复制 driver）；assembly 选型正确（蒸馏 per-sample）；guard 矩阵无新副本；**不引入 server-client/多 LoRA 栈**（§2.4 守护：迁移后 grep 复核 dev 仍只从 `async_rl` 导入 `context_manager`/`streaming_driver`/`types`/`weight_sync`，无 `native_tq`/`pipeline`/`workers`/`scheduler`/`data_plane`/`*_sampler_tq`）。
4. **盲点狩猎**：迁移侧 oracle 是「跑起来不报错」易假绿 → rigor 偏向**集合/数值 parity**（迁移前后 consume 集合一致），而非只验能跑。

---

## 6. 验证计划（作者阶段 AST-only；harness 用完即删；正式 UT 延后到用户要求）

1. AST 全触及文件；真实 import（含新建 `gkd_async.py`）；ruff 新增行零违规、不新增 C901。
2. **control-plane harness（ephemeral `/tmp`，Fake teacher+sampler + 真实 StreamingDriver + 真实 RLContextManager + 真实 mixin `_drive`）**：蒸馏 `lmbda==1.0` staleness=1 跑通；consume 集合 parity vs 迁移前；OPSD/MOPD 继承跑通；drain/reached_max/短流；`lmbda<1` 不误路由。
3. 每条「不应发生 X」断言配「X 会被检测到」的 control（防空跑假绿）。

---

## 7. 施工清单（2 个签收 checkpoint，增量推进）

### ckpt-1：蒸馏族 `lmbda==1.0` async → StreamingGKDLoop@staleness=1
- [ ] 1-1 新建 `swift/dev/recipe/gkd_async.py`：`StreamingGKDLoop(StreamingLoopMixin, GKDLoop)`——assembly=`_assembly_ready_ppo`，`_consume_async_samples` 复用 GKD 现有 teacher-score+train，`_streaming_unit_size`=`train_batch_size*ga`；构造跑 `_check_streaming_config`(staleness=1) + `_init_streaming`(run_id='run_gkd')。确认 OPSD/MOPD 三层继承叠 mixin 后 MRO 正确（子类只覆盖 teacher 信号、`_drive`/`_consume_async_samples`/`_extra_step_metrics` 不错位）。
- [ ] 1-2 `GKDLoop.fit` 重构为经 `_drive`（镜像 `GRPOLoop.fit`）：base `GKDLoop._drive`=`_run_sync`（sync-only）；`StreamingGKDLoop` 用 mixin `_drive` 覆盖。删 `_run_async` 留到 ckpt-2（先确认路由已把 async 引到 StreamingGKDLoop）。
- [ ] 1-3 路由 run_gkd.py/run_opsd.py/run_mopd.py：`one_step_off`+`lmbda==1.0` → `StreamingGKDLoop`(staleness=1, weight_sync_strategy/allow_partial_rollout/adapter_name)；否则 → base `GKDLoop`(`async_generate=False`)。fully_async 蒸馏仍拒。
- [ ] 1-4 validate：蒸馏 `lmbda==1.0` async 走流式后放开 M12 多轮拒斥（per-sample submit_sample→per-episode 多轮）；蒸馏 sync（阻塞 `_generate` 单轮）仍拒 max_turns。更新过时注释（现说「蒸馏仍 overlap 单批」）。
- [ ] 1-5 harness：蒸馏 `lmbda==1.0` staleness=1 跑通 + consume 集合 parity + OPSD/MOPD 继承 + `lmbda<1` 不误路由 + drain/reached_max；用完删。
- [ ] 1-6 冷审（§5 的 1/2/3）+ 汇报 parity，**请签收**。

### ckpt-2：退役 async-only 死代码（ckpt-1 签收后执行删除，D6）
- [ ] 2-1 删 GRPO/PPO base `_run_async`/`_submit_generation`/`_collect_generation` + `GKDLoop._run_async`；GRPO/PPO base `_drive` 收敛为 sync-only（删 async 分支）。
- [ ] 2-2 删 `overlap_rollout_batches` + 文档引用更新（rollout_config.py:89、validate.py:242/251/280/364）。
- [ ] 2-3 删 `submit_generate`/`collect_generate`/`cancel_generate`/`GenerationHandle` + 多轮 guard；删前重跑 §4 grep 复核归零。
- [ ] 2-4 全触及文件 AST + import + ruff；标记 base-loop/GKD-async 旧测试为 UT 阶段更新。
- [ ] 2-5 冷审（§5 全项含退役审计 + §2.4 server-client 守护）+ 汇报，**请签收 Phase 4**。

---

## 8. 非目标（防范围蔓延）
- **sync 迁流式 / staleness=0 单旋钮统一**（Option A 裁定：sync 保留 base，verl 对标 + colocate 约束 + selectable≠统一）。
- **RFT 迁流式**（sync-only，保留 `RFTLoop.fit`）。
- **GRPO/PPO sync 迁流式**（保留 base `_run_sync`）。
- `_generate`/`finish_generate`/`sync_weights`/`PromptBatchScheduler` 退役（sync 仍用）。
- 借用 twinkle server-client / TransferQueue 多 LoRA 栈做单 LoRA/全参训练（§2.4）。
- verl 式 RolloutReplica 弹性增删整副本 + server router 负载均衡；delta / 稀疏 diff 权重传输（§2.4 已知局限，属 twinkle 下沉候选）。
- 性能实测 / GPU 利用率 / 多轮尾巴消除对比（Phase 5）。
- 进程分片 / 协程重写 / 动态资源调度 / verl 式 Ray actor 队列（交接文档 §13 否决）。
- 正式端到端 UT（延后到用户要求，走 testcase-planning）。

---

## 9. Definition of Done
**Feature DoD**：
- [ ] 子 plan 动手前经用户确认（Option A 已裁）。
- [ ] ckpt-1~2 全实现，作者阶段 AST 验证。
- [ ] 蒸馏 async 迁移 parity 已证（consume 集合一致、staleness=1 version 推进正确、OPSD/MOPD 继承正确）。
- [ ] review-mode 冷审对实际代码执行，每个反例实跑通过并配 control。
- [ ] 退役前 §4 消费者逐个 grep 归零，删后全仓复核，文档引用同步。
- [ ] 无静默退化；**sync 路径行为不变**（colocate+disaggregated 都仍可选）；无死代码/半态字段。

**Test DoD**（用户要求 UT 时）：PLAN 表完整、端到端不 mock、反向验证、旧 base-loop/GKD-async 测试更新到迁移后契约。

---

## 10. 风险登记（结论先行）
1. **`GKDLoop.fit` 重构经 `_drive`**：改 sync 入口结构，须保证 base sync 行为逐字不变。缓解：ckpt-1 harness 验 sync 路径 + 冷审不变式。
2. **OPSD/MOPD 继承 + mixin MRO**：三层继承叠 mixin，`_drive`/`_consume_async_samples`/`_extra_step_metrics` 的 MRO 须正确。缓解：冷审 MRO + harness 跑 OPSD/MOPD。
3. **退役遗漏**（Phase 1 教训）：§4 全量枚举 + 删前重跑 grep。
4. **测试滞后**：base-loop/GKD-async 旧测试删除后红——作者阶段 AST-only，UT 阶段统一更新。
