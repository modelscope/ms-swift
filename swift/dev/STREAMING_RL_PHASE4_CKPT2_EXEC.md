# Phase 4 · ckpt-2 执行子计划 — 退役 async-only 死代码（含决策点）

> 本文件 LOCAL-ONLY，不 stage/commit。父计划见 `STREAMING_RL_PHASE4_PLAN.md` §7 ckpt-2 / §2.2 / §4 / §9。
> 前置：ckpt-1（蒸馏族流式迁移 + Option B 根因修复）已签收。D6「先确认归零后删」：删前重跑 grep 复核。

## 0. 一句话目标
删掉批粒度 1-batch-lookahead async 的全部死代码（`_run_async`/`_submit_generation`/`_collect_generation`/`overlap_rollout_batches` + rollout 层 `submit_generate`/`collect_generate`/`cancel_generate`/`GenerationHandle`），base `_drive` 收敛为 sync-only；sync 路径行为逐字不变。

## 1. 归零复核（已 grep 坐实，删前再跑一次）
- 所有 overlap regime（`one_step_off`/`fully_async`）经路由 → `Streaming*Loop`，构造时传 loop 级 `async_generate=False`；base loop 仅在 `async_mode='none'` 到达。
- 全仓无任何 `async_generate=True` 构造（唯一命中是 `process.py:616` 注释 + `_derive_async_mode` 把 **config 级** `RolloutConfig.async_generate` 折进 `async_mode`——那是 CLI 平价配置字段，**保留**，与 loop 级 attr 无关）。
- `RFTLoop(GRPOLoop)` 覆盖 `fit`（bootstrap 轮），从不调 `_drive`/`_run_async`；validate 把 RFT 排除在 `_ASYNC_RLHF_TYPES` 外 → 不受影响。
- `submit_generate`/`collect_generate`/`cancel_generate`/`GenerationHandle` 消费者仅 GRPO/PPO 的 `_submit_generation`/`_collect_generation` + 三个 `_run_async`（GKD 经继承用 GRPO 的）。删上层后归零。
- **保留**（sync + 流式共用）：`_generate`/`_run_sync`/`_rollout_step`/`finish_generate`/`sync_weights`/`_finalize_samples`/`_await_generation`/`_build_sampling_params`/`_build_trajectories`/`_samples_from_responses`/`submit_sample`/`poll_completions`/`collect_sample`/`cancel_sample`/`abort_all_inflight`/`SampleHandle`/`PromptBatchScheduler`/`_train_rollout_batch`/`_consume_async_samples`。

## 2. 删除清单（按文件，line 为当前锚点，执行时按区域实读定位）

### A. `recipe/train_loop.py`
- 删 `overlap_rollout_batches`（58-96，自足，前后 `is_grad_sync_boundary`/`flatten_evalscope_report` 无关）。

### B. `recipe/grpo.py`
- 删 `_submit_generation`（301-317）、`_collect_generation`（319-325）、`_run_async`（1012-…）。
- `_drive`（1052-1063）收敛为 `self._run_sync()`（删 async 分支 + 改 docstring）。
- `_finalize_samples` docstring（274）：删「and the async `_collect_generation`」→ 仅 `_generate`（sync）。

### C. `recipe/run_ppo.py`
- 删 `_submit_generation`（609-617）、`_collect_generation`（619-622）、`_run_async`（672-…）。
- `_drive`（714-725）收敛为 sync-only。

### D. `recipe/run_gkd.py`
- 删 `_run_async`（470-…，GKD 无自己的 `_submit/_collect_generation`，继承 GRPO）。
- `_drive`（509-522）收敛为 sync-only。

### E. `rollout/__init__.py`
- 删 `GenerationHandle`（134-…）、`submit_generate`（516-559，**含多轮 guard 540-543**——流式 `submit_sample`/episode 已支持多轮，guard 随批路径作废）、`collect_generate`（561-575）、`cancel_generate`（577-581）。
- 更新引用这些符号的 docstring/注释：类 docstring(105)、`SampleHandle`(152 "counterpart of GenerationHandle")、`_build_sampling_params`(495)、流式段注释块(583-588 "smallest case of the batch submit_generate above")、`collect_sample`(746 "counterpart of collect_generate")、`abort_all_inflight`(805 "Distinct from cancel_generate")。

### F. 文档引用清理（跨文件）
- `gkd_async.py:9`：docstring 提 ``overlap_rollout_batches`` → 改述为「retired batch-granular 1-batch-lookahead overlap」，不点名已删符号。
- `_streaming_loop.py:348`：提 ``_collect_generation`` → 只留 `_generate`（保留的 sync 批路径）。
- `config/rollout_config.py:89`、`config/validate.py:242/251`（+ 复跑 grep 确认 280/364 若存在）：`overlap_rollout_batches` 交叉引用改述，避免 sphinx 悬挂 `:func:`。

## 3. ★决策点：loop 级 `async_generate` attr 是否一并删除

删掉三个 `_drive` 的 async 分支后，**`self.async_generate` 变为零 reader 的死字段**（唯一读点就是被删的分支；grpo.py:713 仅注释提及）。父计划 §9 DoD 明列「无死代码/**半态字段**」，razor 亦要求删。但父计划 §7 ckpt-2 任务只写「`_drive` 收敛为 sync-only」，未显式列删 attr——这是计划内部的一处 gap，需你裁定。

**选项 1（推荐，彻底 / 合 DoD）**：连同 attr 一起删。额外触及：
- 构造签名 + attr 写：`grpo.py:170,251`、`run_ppo.py:399,429`；`run_gkd.py:311,347`（透传 super）。
- 构造调用点删 `async_generate=False`：`run_grpo.py:574`、`run_ppo.py:249`、`run_gkd.py:222`、`run_opsd.py:226`、`run_mopd.py:216`。
- 三个 Streaming loop 删 `kwargs['async_generate']=False` + 注释：`grpo_async.py:101-103`、`ppo_async.py:116-118`、`gkd_async.py:93-95`（mixin `_drive` 不读它，删后无副作用）。
- `grpo.py:713` 注释改述。
- 端到端一致：此后**唯一真源是 config `async_mode` → 路由选 loop 类**，loop 层不再有 async 旗标。
- 代价：约 +6 文件的机械改动；测试构造点（test_rollout.py:471、test_recipes.py GKDLoop×3/GRPOLoop）均用默认值、未传该 kwarg → 不破。

**选项 2（最小 / 留半态字段）**：只删方法 + 收敛 `_drive` 分支，保留 `async_generate` kwarg/attr（恒 False、无处读）。触及文件少（~5），但**违反 DoD §9「无半态字段」**，且 Streaming loop 里 `kwargs['async_generate']=False` 变成无意义仪式。不推荐。

→ **我按选项 1 执行**，除非你改选 2。

## 4. 验证（作者阶段 AST-only；harness 用完即删；正式 UT 延后）
1. 全触及文件 `ast.parse`。
2. 真实 import 五个 loop 模块 + rollout + train_loop（无悬挂引用 / 无 ImportError）。
3. 删后全仓 grep 复核：`overlap_rollout_batches`/`submit_generate`/`collect_generate`/`cancel_generate`/`GenerationHandle` 零命中（除历史 MD）；若选 1，`async_generate`（loop 级）零命中，config 级 `RolloutConfig.async_generate`/`_derive_async_mode` 仍在。
4. MRO + hook 归属复跑（五个 Streaming loop 的 `_drive`/`_collect_per_sample` 仍只在 mixin）。
5. 控制面 harness 复跑（GRPO/PPO/GKD × 两策略 + 反向验证）确认收敛 `_drive` 未伤流式；用完删。
6. 冷审 §5：不变式（sync 逐字不变、`_generate`/`finish_generate`/`_finalize_samples` 保留链完整）、盲点（删除侧无 oracle → rigor 偏向 grep 归零 + import + sync 路径冷读）。

## 5. 非目标
- 不动 sync 路径任何行为；不动 config 级 `async_generate`/`async_mode`/`_derive_async_mode`；不动 twinkle；不引入 server-client/多 LoRA 栈（§2.4 守护）；不写正式 UT（延后到用户要求）。

## 6. DoD（ckpt-2）
- [ ] §1 归零复核删前重跑通过。
- [ ] §2 A–F 删除 + 文档引用同步，无悬挂 `:func:`/`:meth:`/`:class:`。
- [ ] §3 决策按裁定执行（默认选项 1：无 loop 级半态字段）。
- [ ] §4 验证 1–6 全绿；sync 路径冷读确认逐字不变。
- [ ] 汇报，请签收 Phase 4。
