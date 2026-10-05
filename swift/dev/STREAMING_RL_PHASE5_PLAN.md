# Sync/Streaming 统一决策子计划（Phase 5：调查结论 + 收口方案）

> LOCAL-ONLY：本文件不 stage、不 commit。
> 状态：**待用户 sign-off**。本子计划的核心是一个「该不该做」的决策，结论是**不做全量 staleness=0 统一**，改做小范围诚实收口。

## 0. 触发与用户要求

用户要求：方案必须对**所有 RL 训练 × 所有模型类型 × 所有 sample 类型统一**；可以两个类，但逻辑清晰、复用性强、扩展性强；performance 对标 verl。
本子计划回答：这个「统一」在哪条轴上已经成立、在哪条轴上不该强求，以及真正值得做的收口。

## 1. 结论先行

1. **算法/模型/样本轴：统一已成立，无需新工作。**
2. **sync↔overlap / colocate↔disaggregated 轴：不做 staleness=0 全量统一。** 深挖后确认它有害（见 §3），且 verl 同样分开。
3. **建议只做小范围诚实收口**：修正误导注释（A）+ 修一个真实 robustness 缺口（B）。C（共享 sync driver）边际收益，默认不做。

## 2. 事实基线（四路 Explore 已核实，附 file:line）

### 2.1 算法/模型/样本轴——已经是统一的
- 五个在线算法 GRPO/PPO/GKD/OPSD/MOPD 共享：
  - 一个 `StreamingLoopMixin`（`swift/dev/recipe/_streaming_loop.py:73`）——全部流式接线；
  - 一个 twinkle `StreamingDriver`（`twinkle/src/twinkle_agentic/async_rl/streaming_driver.py:160`）——算法无关；
  - 一个 `RolloutSample` 类型（`swift/dev/rollout/__init__.py:63`）；
  - 一个后端无关、placement 无关的 `RolloutEngine`（`rollout/__init__.py:358`）。
- 算法差异压到 3 个 hook + 2 条 assembly 规则：`_consume_async_samples` / `_streaming_unit_size` / `_streaming_step_delta`；`_assembly_ready_grpo`（完整 group）/ `_assembly_ready_ppo`（逐样本），由 `_assembly_ready` 一个 hook 选。
- 模型/后端/单轮多轮差异全下沉 rollout 层与 builders：`samples_from_responses`（token 路）/`_message_only_samples`（文本 teacher）/`trajectory_to_rollout_sample`（多轮）；vLLM/SGLang 在 rollout 层与 driver **无分支**（`rollout/__init__.py:16-19`）。
- 结论：**driver 与 mixin 内不存在 per-algorithm / per-model / per-sample-type 分支**。用户要的统一在这条轴上已兑现。

### 2.2 sync 路径现状（未统一的那套）
- 三份 `_run_sync`：`grpo.py:967`（GRPO）、`run_gkd.py:449`（GKD/OPSD/MOPD）、`run_ppo.py:632`（PPO）；RFT 用自带 `fit`（`run_rft.py:276`），不经 `_drive`/scheduler。
- 都基于 `PromptBatchScheduler`（`train_loop.py:95`）+ 阻塞 `_generate`（`sync_weights → generate → finish_generate → _finalize_samples`）。
- `_drive` 已是干净模板 hook（`grpo.py:998`/`run_gkd.py:486`/`run_ppo.py:671`），base 仅 `self._run_sync()`。

### 2.3 twinkle 对 staleness=0 的真实支持度
- **gate 层面支持**：`RLContextManager` gate（`context_manager.py:177-181`）在 0 处良定义、0 是默认、有单测（`test_async_rl_native_tq.py:266-278`）。S=0 时最多 1 个 live window，靠 `on_partition_cleared` 才重开。
- **driver 层面有三个硬伤**：
  1. admission 循环**不受 window 可用性约束**（只有 open window 受约束，`streaming_driver.py:457-466` 丢弃返回值）。S=0 的「同步」= 一次一个 publish cycle，**不是**「训练时停生成」。
  2. **S=0 每次 publish 会丢弃所有未训 v0 样本**（在飞 + 缓冲，两处 guard 都是 `> 0`：`:383`、`:484`），算力被浪费而非取消。
  3. **没有 drain-before-publish 接缝**；`StreamingDriver` **全仓零测试**。

### 2.4 colocate 是真障碍
- 设备 hand-over 在 `swift/dev/recipe/_colocate.py:20` `ColocateHandover`，**两阶段**：`enter`（wake sampler weights → sync → offload trainer → wake KV）/ `exit`（sleep sampler → reload trainer）。
- `WeightSyncStrategy.publish` 是**单阶段**（`weight_sync.py:45-51`），driver **没有** pre-admission / post-collect / drain 接缝能放 `exit`。
- colocate 要求 trainer 与 sampler **独占设备轮流**（`manager.py:52-66`、`model/base.py:268-276`），与 driver 的**单线程交错 + 非阻塞后台生成**根本冲突。
- twinkle 明确拒绝拥有 memory schedule（「the caller also owns the memory schedule」`manager.py:52`），且「transport 是部署属性不是 strategy」（`weight_sync.py:14-17`、`:158-162` 的 `'nccl'` tombstone）。
- 前序 `STREAMING_RL_PHASE4_PLAN.md:12` 已推演：S=0 首遍不 publish，colocate admission 会在 sampler 未唤醒时提交 → 崩。
- **colocate 是 sync 默认部署**：`cli/rlhf.py:86-89` 对所有在线类型默认 `vllm_mode='colocate'`。

### 2.5 现有 gate（若真要放开需改的点，供参考）
`validate.py:331-335`（overlap 强制 disaggregated）、`:308-310`+`_reject_fully_async_only_knobs`（`none` 下拒绝一切流式旋钮含 staleness=0）、`:418-429`（one_step_off 钉 1 / fully_async 需 ≥1）、`_streaming_loop.py:110-112`（构造期 `max_staleness>=1`）。

## 3. 为什么不做 staleness=0 全量统一（决策依据）

| 判据 | 结论 |
|---|---|
| 能否退役 `_run_sync` | **不能**：colocate（默认 sync 部署）搬不上 driver，仍需 `_run_sync` |
| 净效果 | 多出「disaggregated-sync-on-driver」第三种跑法，**更复杂** |
| 性能 | S=0 每次 publish 丢弃未训样本、无 drain，**比干净 lockstep 更差** |
| 风险 | 需给**零测试**的 twinkle `StreamingDriver` 核心控制流动手术（加 drain + 设备 acquire/release 接缝） |
| 框架一致性 | 逆 twinkle 明确设计（placement=部署属性、caller 拥有 memory schedule） |
| verl 对标 | verl **同样分开**（colocate hybrid engine 同步环 vs disaggregated/server 异步环），对标**不要求**此统一 |

→ **不做**。sync/overlap 分界对应物理真实的轴，保留两套是正确设计。

## 4. 建议执行的收口（低风险，逐项可独立 sign-off）

### A. 修正误导性注释（纯文档，零行为变更）
把「later phase 会把 sync 迁到 staleness=0 / 0=同步流」的承诺改成如实描述：
- `_streaming_loop.py:15-18`（模块 docstring）、`:94-97` + `:110-112`（`_check_streaming_config` 的 guard 文案）、`:31-32`。
- `config/rollout_config.py:83-84`（`'none'` 描述可保留，但确认不承诺 driver 化）。
- `twinkle/.../streaming_driver.py:11-14`（driver docstring「0=同步」）：**这是 twinkle 侧**，改动需谨慎——twinkle 的 driver 本就支持 S=0（gate 良定义），只是 dev 不用它跑 S=0。建议**不改 twinkle docstring**（它对 twinkle 自身是准确的），只改 dev 侧承诺。
- 新文案要点：`StreamingDriver` 在 dev 只服务 **overlap regime（staleness≥1，disaggregated-only）**；`async_mode='none'` 走固定批 `_run_sync`（colocate + disaggregated 都支持）；staleness=0 在 dev 不启用。

### B. 修 PPO `_run_sync` 缺失的 try/finally（真实 robustness 缺口）
- `run_ppo.py:637-645`：`sync_weights()` 后 `generate()` **无 try/finally**，生成异常会跳过 `finish_generate()`，把设备留在 hand-over 状态（sampler 醒着、trainer offload 未恢复）。GRPO 的 `_generate`（`grpo.py:283-291`）有 try/finally，PPO 没有。
- 根因修复：把 PPO 的 sync 生成也收敛成「`sync_weights → try: generate finally: finish_generate`」，与 GRPO 对齐。**最佳做法**：让 PPO 复用 GRPO 同款 `_generate` 语义，而不是各写一份（消除接缝不一致）。评估两方案：
  - B1：PPO `_run_sync` 内联加 try/finally（最小改动）。
  - B2：把阻塞生成 bracket 抽成 rollout 层或基类的一个 `_blocking_generate(prompts, extras)`，GRPO/PPO 共用（更彻底的去重，符合"通用能力下沉"）。**推荐 B2**，因为 hand-over bracket 本就是 placement 通用逻辑，不该在各 loop 各抄一份。

### C.（可选，默认不做）抽共享同步 driver
三份 `_run_sync` 脚手架去重成一个基类模板 + `_sync_batch_action(prompt_indices, round_index)` hook。但它们本就只有 3-5 行、per-batch 动作各异（GRPO 有 DAPO resample/ReMax、GKD 有 lmbda coin-flip + off-policy dataset 轮、PPO 逐样本），抽取后 hook 化收益边际。**倾向不做**，除非用户明确要。

## 5. Review-mode 计划（执行 A/B 时）
- A：逐处确认改的是注释/docstring，**无代码语义变更**；改后 grep 确认没有残留"later phase 迁移 sync"的悬挂承诺；确认没有把 twinkle 侧改坏。
- B：invariants——`finish_generate` 在 generate 抛异常时**必被调用**（设备必复位）；B2 下 GRPO/PPO 走同一 bracket，行为逐字等价。反例：mock `rollout.generate` 抛异常，断言 `finish_generate` 仍被调用（B1/B2 都要）。
- 复用/分层审计：B2 的 bracket 放哪层——rollout engine（`RolloutEngine` 已有 `generate`/`sync_weights`/`finish_generate`）还是 recipe 基类？倾向 rollout engine 提供一个 `blocking_generate(...)` context/helper，因为 hand-over 是 rollout 层职责（`grpo.py:14-19` 已述"weight-sync delegated to rollout"）。

## 6. 验证计划
- A：纯注释——AST parse 触及文件；grep 复核无悬挂承诺；无需跑测。
- B：AST + 真实 import；e2e（用户要 UT 时）——真实模板 tokenizer-only + 脚本化 sampler + 假 tool_manager，驱动 GRPO 与 PPO 的 sync `_drive`，含「generate 抛异常」反例，断言 `finish_generate` 被调用、设备状态复位。B2 额外断言 GRPO/PPO 复用同一 bracket（无第二份内联）。
- 全仓 grep：确认 sync 路径消费者未受影响。

## 7. Definition of Done
- [ ] 用户对「不做 staleness=0 全量统一」的结论 sign-off。
- [ ] 用户选定执行 A / B（B1 或 B2）/ C 的哪些项。
- [ ] 选定项实现 + AST 校验；review-mode 反例执行通过。
- [ ] （若做 B）e2e 反例证明异常路径下 `finish_generate` 必被调用。
- [ ] 无框架逆向改动；twinkle `StreamingDriver` 核心控制流**不动**。

## 8. 非目标
- 不给 `StreamingDriver` 加 drain / 设备接缝。
- 不把 colocate hand-over 塞进 `WeightSyncStrategy`。
- 不放开 staleness=0 的任何 gate（`validate.py` / `_check_streaming_config`）。
- 不动 twinkle 核心（除非用户明确批准）。
