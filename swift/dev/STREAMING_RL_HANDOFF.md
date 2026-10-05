# 交接文档：Per-sample 流式 RL Driver（给无上下文的 qodercli 接手者）

> 本文件是**自包含**交接说明。你（qodercli）没有之前的对话上下文，读完这一份即可接手。
> 动手前**必读**项目规范 skill：`.qoder/skills/dev-module-authoring/SKILL.md`（分层/落位/复用/fail-loudly/剃刀/AST-only 测试纪律）与 `llm-terminology`（中文术语）。
> 写/审测试时必读 `.qoder/skills/testcase-planning/SKILL.md`。
> 同目录相关 RL 文档：`RL_PLAN.md`、`RL_PROGRESS.md`、`RL_MIGRATION.md`、`VERL_COMPARISON.md`。
> 术语约定：回复用标准中文 LLM 术语，但 staleness / publish / abort / resume / partial rollout / version-span / pin / mixin / MRO / window / backpressure / group / straggler / lcm / gcd / drain / episode 等保留英文；**禁止内部编号/代号出现在给用户的回复里**（本文件内部为索引保留 Phase 编号等）。

---

## 0. 一句话现状

大任务是「把 dev 侧 RL driver 从 **batch 粒度锁步** 改为 **per-sample 流式**」：所有 sample 单独 rollout / 工具调用 / API 调用，各组件（sampler / env pool / tool / api）各自提供最大并发控制、各司其职互不干扰，sync/async 不再是两条代码路径而是 `max_staleness` 旋钮上的取值（0=sync、1=one_step_off、>1=fully_async）。用户明确「不必对齐 veRL，把自己方案的效果与性能做到最好」。

- **Phase 1（流式 driver 核心 + 单轮 async GRPO/PPO）已完成并经用户「可以」签收。**
- **Phase 2（多轮 async）本会话已把代码写完（seams 1-3）并通过 AST / import / ruff / ephemeral harness 验证，尚未请用户签收。**
- **Phase 2 的 seam 4（version-span 跨轮上浮）经架构定案延后到 Phase 3**（理由见 §4）。
- **正式端到端测试尚未写**（按纪律延后到用户明确要求，届时走 testcase-planning）。
- **下一步：请用户签收 Phase 2 → 进 Phase 3（async 指标 + version-span）。**

**全部改动都在工作区未提交（working tree）**，没有一个 commit。`git status -s` 可见。注意 `twinkle/` 整个目录在 ms-swift 仓里是 **untracked**（`?? twinkle/`），但它是 editable 安装的活源码（`twinkle/src/...`），我对它的改动**已落盘生效**，你直接编辑同一批磁盘文件即可。

---

## 1. 环境与运行铁律（踩过的坑，务必先读）

1. **解释器钉 `/usr/local/bin/python`（3.12.13）**。conda `(base)` 是 3.14，**没装 twinkle**，裸 `pytest`/`python` 落到它会导致进程级 `ModuleNotFoundError: No module named 'twinkle'`（全有或全无、跨进程间歇）。跑前 `which python` 确认，或 `head -1 $(which pytest)` 反查。
2. **跑 dev 测试用 `pytest` 控制台脚本，不要 `python -m pytest`**：仓库根有 `twinkle/` 命名空间目录，`python -m pytest` 把 CWD 插到 `sys.path[0]` 会遮蔽真包。若必须用 `-m pytest`，加 `--import-mode=importlib`。完整命令模板：`cd 仓库根 && CUDA_VISIBLE_DEVICES=<空闲卡> /usr/local/bin/python -m pytest <target> -m slow --import-mode=importlib -p no:cacheprovider`。
3. **终端 banner 噪声**：每条命令输出前有一段 DSW/PAI 挂载表格。过滤：`grep -vE "DSW|PAI|BMCPFS|mount_path|dataset_info|commit_to_image|Mounted datasets|Welcome to|^│|^╭|^├|^╰|^ ____|^\|"`。判断前置命令是否真失败要看它自身输出，**不要凭管道末端 grep 的退出码**（grep 滤光所有行时 exit 1 会被误读为失败）。
4. **AST-only 验证纪律**：写框架/正式代码阶段只用 `ast.parse` 校验语法，**不擅自跑 pytest/lint/compileall**；用户明确要求补测试时才写完整端到端测试（走 testcase-planning）。ruff 可跑（本会话跑了），配置见 `pyproject.toml`：`select=["B","C","E","F","W","I"]`、`ignore=["F401","F403","F405","F821","E251"]`、`line-length=120`、mccabe 阈值 10（C901 在 select 内但**项目实际不 gate**：base HEAD 的 `validate.py` 已有 6 个已提交 C901）。
5. **GPU**：8 卡（单卡约 143GB）。跑前 `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` 确认空闲卡，用 `CUDA_VISIBLE_DEVICES=` 指定。
6. **后端等价铁律**：vLLM / SGLang 对称，streaming driver 不含 backend 分支。**sglang 本机未装**，涉及 sglang 只能静态核实，不能真跑。
7. **ephemeral harness 用完即删**：自写验证脚本放 `/tmp`（天然不入仓）或用完 `rm`。spawn 型脚本加 `if __name__ == '__main__'` 守卫；加载非包路径脚本用 `importlib.util.spec_from_file_location`。
8. **下结论前先核实源码**：涉及框架/库行为要定位到确切代码（读到实现、必要时实跑）再下结论，避免误判后反转。子代理/grep 结论须回读源码坐实。工作区根路径 Grep 偶发假阴性，须用限定子目录或 `Bash grep -rln` 复核。
9. **编辑工具报「save failed」先回读目标区间**再决定动作，勿盲目重放。

---

## 2. 架构：核心控制流与分层

### 2.1 StreamingDriver 控制流（Phase 1 已建，twinkle 层算法无关）

```
prompt_stream → [per-sample admit] → engine (continuous batching)
                                         ↓ (as-completed poll)
                                    ready_buffer (带 policy_version)
                                         ↓ (assembly_ready?)
                                    consume (train, 复用 _train_rollout_batch)
                                         ↓ (每 K 个 optimizer step)
                                    publish (weight sync: adapter_snapshot 或 in_place+abort/resume)
```

- **Admission**：逐 trajectory 准入（一条 trajectory 一个 submission_id），完成一条立刻从 prompt_stream 回填下一条。两道门控：① staleness gate（in-flight sample 的 version lag ≤ max_staleness）；② backpressure（`in_flight + ready_count < buffer_depth`）。**并发上界由 driver backpressure 唯一封顶**，引擎/env pool/tool/api 各自自限，driver 不另设 pool。
- **Collection**：非阻塞 poll（as-completed），完成的移入 ready_buffer，附带 policy_version。**staleness 双卡点**：completion 一被 fetch 就检查 `current - policy.version > max_staleness` 并在 collect 处丢弃；publish 后的 `_drop_stale` 只扫已入 buffer 后老化的记录。（教训：任何「上界」门控必须覆盖全部持有该属性的容器——in-flight 与 ready 两处都要查。）
- **Assembly + Consume**：GRPO/RFT 拉完整 group（prompt × num_generations 齐）、PPO/GKD 拉单 sample，凑满 `train_batch_size × ga` 整数倍后 consume。GRPO 组装 pull 量子 = `lcm(need_rows, num_gen) / num_gen` 个 group，保证既是精确整数倍又由完整 group 组成（`compute_advantages` 要求「每 num_generations 连续块=一个 group」，`n % num_generations != 0` 会 raise）。余数退回 buffer 不丢弃。
- **Publish**：每 `parameter_sync_step`（K，默认 1）个 optimizer step 触发一次 weight sync。
- **pin 生命周期单次释放**：`acquire_rollout_policy` 在准入加 refcount，`release_rollout_policy` **仅在 collect 处释放**（`release_rollout_policy` 在 count<=0 会 raise，故 buffer 退出/stale 丢弃处**不可**再 release）。
- **Drain**：prompt_stream 耗尽后停止准入，等 in-flight 全部 collect 或 cancel，最后 consume 剩余 ready。
- **Straggler**：trainer 永远训「已就绪的」，不等慢的。

### 2.2 控制面 / 数据面分层（Phase 2 的核心架构判据，务必理解）

- **version / policy 生命周期是控制面概念**：`RLContextManager.acquire_rollout_policy / on_partition_trained`（版本自增）+ StreamingDriver 的 pin / lag 计算。
- **核心 `sampler.sample()` 与 twinkle `MultiTurnRollout` 引擎是数据面**，**刻意版本无关**，且被 sync 多轮 / 教师蒸馏 / sampling-inference 共用。
- **把 version 塞进引擎会反转分层**——这是 Phase 2 拒绝「per-turn version 跟踪进引擎」方案的根本理由（详见 §4）。

### 2.3 复用的既有原语（一律不动）

`RLContextManager`（version/pinning/staleness gate）、`WeightSyncStrategy`（publish：`InPlaceWeightSync` / `AdapterSnapshot`）、`PartialRolloutMixin`（abort/resume）、`_train_rollout_batch`（GA→step→replay）、`_run_micro_step`。

---

## 3. Plan 全文位置与各 Phase 状态

**Plan 文件（只读，禁止编辑）**：
`/root/.config/Qoder/425f308ab851fa8e84da4df0a83ffb968678e145b275d091387556e38da8323b/SharedClientCache/cache/plans/Per-sample_Streaming_RL_Driver_8c0d73ae.md`（178 行）

| Phase | 范围 | 状态 |
|-------|------|------|
| **Phase 1** | StreamingDriver 核心 + 单轮 async GRPO/PPO + 退役旧 batch 骨架 | ✅ 完成，**用户已签收** |
| **Phase 2** | 多轮 async：放开 validate M12、per-episode 准入、allow_partial_rollout 透逐轮、~~version-span 跨轮上浮~~ | ✅ 代码完成（seams 1-3），**待用户签收**；seam 4 version-span **延到 Phase 3** |
| **Phase 3** | async 指标：StreamingDriver 计时器/计数器 + RolloutSample version-span 字段 + tracking.py 发射 | ⬜ 未开始（**下一步**） |
| **Phase 4** | sync 路径迁移（staleness=0）：GRPO/PPO/RFT/GKD `_run_sync` 改 StreamingDriver；蒸馏族流式迁移；退役固定批 generate() | ⬜ 未开始 |
| **Phase 5** | 性能实测：fully_async 多卡真跑、多轮 async 吞吐/straggler/GPU 利用率、sync streaming vs 旧固定批、parameter_sync_step K 调优 | ⬜ 未开始（写完再处理） |

**Plan 纪律**：每 Phase 签收后才进下一 Phase；**不编辑 plan 文件**；Phase 1-4 作者阶段 AST 验证，测试延后到用户要求时按 testcase-planning 出 PLAN 表。

---

## 4. Phase 2 架构定案：version-span 为什么延到 Phase 3（关键决策，勿推翻）

Plan 原本把「per-episode version-span 跨轮上浮」列为 Phase 2 的 seam 4，且 plan L123 假设 twinkle `RolloutOutput` 已带 `rollout_policy_versions` / `policy_version_span` 供 dev 上浮。

**全仓核实证伪了这个前提**：`rollout_policy_versions` / `policy_version_span` 这两个字段**只存在于无关的 legacy TransferQueue 运行时**（`twinkle/src/twinkle_agentic/async_rl/` 下的 `vllm_sampler_tq.py`、`types.py`、`workers.py`、`pipeline.py`——即 `AsyncMultiLoraGRPOPipeline`）。**dev streaming driver 根本不走它**。dev streaming 路径的 `SampledSequence` / `SampleResponse` / `submit_generation` / `get_generation_status` / `collect_generation` 以及 twinkle `MultiTurnRollout` 输出的 trajectory（`ledger.merge`）**全部无 version 字段**——引擎完全版本无关。

**据此定案（用户把决策权交给架构判断，问「哪种复用性最大、架构最合理」）**：

- version 是纯**控制面**概念。把 per-turn version 跟踪塞进共享引擎（twinkle `MultiTurnRollout` / `sampler`）会**反转 data-plane/control-plane 分层**，且爆炸半径覆盖所有多轮消费者（sync 多轮 / 教师蒸馏 / sampling-inference 共用同一引擎）——**复用性最差、最不合理，否决**。
- 在 dev 层现在建半态 version-span 字段（无消费者）——**留死字段，否决**。
- **采纳**：Phase 2 只做 seams 1-3；**version-span 延到 Phase 3**，做成 **driver 端 strategy-aware 标量**（引擎不入版本感知）：
  - `adapter_snapshot`：整条 episode pin 到一个 frozen `adapter_path`，逐轮同版本，**span ≡ 0**。
  - `in_place`（`adapter_path=None`）：逐轮采活权重，publish 时 abort+resume，episode **跨版本**，span = `collect 时 current_version − admit_version`（即 driver 已经算出的 staleness lag）。
- 这与 plan L153（Phase 3 才放「RolloutSample version-span 字段」）**自洽**。
- **staleness 锚点**：`policy_version`（准入版本）是最保守/最老锚点，多轮 episode 也正确；GRPO 的 TIS 用逐 token old_logps（生成时记录），与 version-span 无关。故 **version-span 在 Phase 2 无正确性消费者**，唯一用途是 Phase 3 指标。Phase 2 保留 `policy_version` 锤点即可。

---

## 5. Phase 2 已完成的代码改动（精确清单，全部未提交）

> 定位以**符号名**为准（行号会漂移）。下列 4 个文件是 Phase 2 触及的全部文件。

### 5.1 `twinkle/src/twinkle_agentic/rollout/multi_turn.py`（数据面引擎，seam 3 落点）

- `_resolve_call`：新增 **gated** 的 `allow_partial_rollout` 透传——
  ```python
  if self.sampler is not None and kwargs.get('allow_partial_rollout', False):
      adapter_kwargs['allow_partial_rollout'] = True
  ```
  仅当有 sampler 驱动逐轮时才透传（API/teacher 后端无 in-place weight sync 可 resume，且其 callback 不吃这个 kwarg）。`adapter_kwargs` 进 ctx，最终由 `_default_response_callback` 的 `sampler.sample([input_feature], sampling_params=..., **adapter_kwargs)` 逐轮传入。
- **链路闭合已核实**：dev `rollout_kwargs` → `self.rollout(trajectories, sampling_params=sp, **rollout_kwargs)` → base `Rollout.__call__(**kwargs)` → `_resolve_call(kwargs, n)` → `adapter_kwargs` → 逐轮 `sampler.sample(**adapter_kwargs)` → 核心 `_run_partial_rollout`（`PartialRolloutMixin`），使 in_place publish 的 `abort_all_inflight` 能中断+resume 逐轮。
- 引擎输出 trajectory（`ledger.merge`）**未改**（不上浮 version，符合 §4 决策）。

### 5.2 `swift/dev/rollout/multi_turn.py`（dev 薄包装层，seam 3）

- `MultiTurnRollout.generate` 签名新增 `allow_partial_rollout: bool = False`。
- `rollout_kwargs` 构造（**复杂度中性写法**，见 §7 冷审发现2）：
  ```python
  rollout_kwargs = {'allow_partial_rollout': allow_partial_rollout,
                    **({'adapter_path': adapter_path} if adapter_path else {})}
  ```
  `adapter_path` 为 falsy 时**整个省略**（无 LoRA 的 sampler 必须看到与以往相同的调用）；`allow_partial_rollout` **总是传**（`False` 在 `_resolve_call` 里与省略等价，因为它 gate 在真值上）。
- `_generate_with_envs` 签名改为 `(self, trajectories, sampling_params, **rollout_kwargs)`，内部把 `**rollout_kwargs` 逐字转发给每个 per-episode 引擎调用（sandbox 路径也 pin 同样权重、开同样 abort/resume）。该方法已是 per-episode `ThreadPoolExecutor`（每 episode 独立 leased env，`workers=max(1, len(env_pool))`）。
- `trajectory_to_rollout_sample` **未改**（不上浮 version）。

### 5.3 `swift/dev/rollout/__init__.py`（RolloutEngine，seam 2 = per-episode 准入）

- imports 新增 `import threading` 与 `from concurrent.futures import Future`。
- `RolloutEngine.__init__` 新增 `self._episode_futures: Dict[str, Future] = {}`（**只有 driver 线程改这个 dict**：submit 加、collect/cancel pop；episode 线程只碰自己的 Future）。
- `submit_sample`：开头新增多轮分支——`if self._multi_turn is not None: return self._submit_episode(...)`（透传 prompt_idx / sampling_params / prompt_extras / force_logprobs / adapter_path / allow_partial_rollout / policy_version）。单轮路径不变（走 `sampler.submit_generation`）。
- **新增 `_submit_episode`**（per-episode 流式准入）：建 `submission_id` + `Future`，**先注册 `self._episode_futures[submission_id] = future` 再起线程**（无注册竞态），在 **daemon 线程**里跑单 episode 的阻塞 `self._multi_turn.generate([prompt], num_samples=1, ..., adapter_path=, allow_partial_rollout=)`，`set_result(samples)` / `except Exception as exc: set_exception(exc)`（线程绝不能不 resolve 就死），立即返回 `SampleHandle`。**并发由 driver backpressure 唯一封顶**，本层不设 pool（线程绝大多数时间 parked 在 GPU/tool I/O，每条 in-flight episode 一个线程很便宜）。
- `poll_completions`：循环内**先查 `self._episode_futures.get(handle.submission_id)`**——若是 episode：`not future.done()` → continue（仍在跑）；`future.exception()` 非 None → `raise RuntimeError(f'streaming multi-turn episode {submission_id} failed: {error}') from error`（fail-loudly，镜像单轮契约）；否则 `completed.append(handle)`。非 episode 才走 `sampler.get_generation_status`。
- `collect_sample`：开头**先 `future = self._episode_futures.pop(handle.submission_id, None)`**——若是 episode：`samples = future.result()`（失败则在此 re-raise，fail-loudly）；`len(samples) != 1` → raise「expected exactly 1 sample for one multi-turn episode ... got N」；stamp `sample.prompt_id = str(handle.prompt_idx)`、`sample.policy_version = handle.policy_version`；return。非 episode 才走 `_await_generation` + `_samples_from_responses`。
- `cancel_sample`：`if self._episode_futures.pop(handle.submission_id, None) is not None: return`（**abandon**：episode 无 sampler submission 可取消，daemon 线程自己跑完、结果丢弃）。非 episode 才 `sampler.cancel_generation`。
- `close()`：末尾新增 `self._episode_futures.clear()`（drop 引用；daemon 线程随进程退出）。
- **保留** `submit_generate` 的多轮 guard（`raise RuntimeError('async_generate does not support the multi-turn rollout ...')`）——见 §6 plan 偏离1。

### 5.4 `swift/dev/config/validate.py`（seam 1 = 放开 M12）

- `_check_async_mode` 的 M12 拒斥**从「任何 max_turns 都拒」改为「只拒蒸馏族」**：
  ```python
  if rlhf_config.max_turns is not None and rlhf_type in {'gkd', 'opsd', 'mopd'}:
      raise ValueError(... 蒸馏族走 submit_generate 单批重叠、只驱动单轮 sampler、无 per-episode 多轮准入；
                       GRPO/PPO 走 per-sample streaming driver，每条 episode 独立线程准入 ...)
  ```
  GRPO/PPO 现在**允许** max_turns + async（走 streaming `submit_sample` → `_submit_episode`）。
- 更新了 `_check_async_mode` docstring 里过时的一条（原说「multi-turn 不能 pre-submit」，改为「dynamic_sample 仍不可；multi-turn 现在可 per-episode 准入（GRPO/PPO），蒸馏族单批重叠仍拒 max_turns 直到其流式迁移」）。
- **保留** `dynamic_sample`（L332 区）与 `advantage_estimator == 'remax'`（L337 区）的拒斥。
- GRPO/PPO 仍走 `_check_streaming_publication`（strategy 配对 + parameter_sync_step>=1 + one_step_off 钉 staleness=1），**无需改**（多轮不改 strategy 配对）。
- **已核实 `_check_streaming_config`（在 `_streaming_loop.py`）无任何多轮拒斥**，且 `_streaming_loop.py` / `grpo_async.py` / `ppo_async.py` 全文无 `max_turns` / `multi_turn` 引用——**streaming driver/loop 对多轮完全无感**，路由只在 `RolloutEngine.submit_sample` 内部发生。`configure_multi_turn` 由基类 `run_grpo.py` 配置到共享 rollout engine，async loop 通过继承拿到。

---

## 6. Plan 偏离（已 flag，接手者须知，勿盲目「修回」plan 原文）

1. **`submit_generate` 多轮 guard 保留**（plan L81/L116 要求移除）。理由：`submit_generate` 是**旧 batch async API**（被 `grpo.py::_submit_generation` / `run_ppo.py` 经 `overlap_rollout_batches` 消费，仅蒸馏族 one_step_off 用），**不是 Phase 2 的 streaming path**。移除它而不给 batch 路径接多轮引擎，会**静默退化成单轮**。**正确的移除时机是 Phase 4**（蒸馏族迁到 streaming + per-episode 多轮支持后）。
2. **version-span 从 Phase 2 移到 Phase 3**（plan L123/L148 列为 Phase 2 seam 4）。理由见 §4（plan 前提被证伪 + 控制面/数据面分层）。

---

## 7. 验证状态（本会话已做）+ 冷审发现

### 7.1 已通过

- **AST**：4 个触及文件 `ast.parse` 全过。
- **真实 import**：`swift.dev.rollout`、`swift.dev.rollout.multi_turn`、`swift.dev.config.validate`、`twinkle_agentic.rollout.multi_turn` 全 OK。
- **ruff（项目 select B,C,E,F,W,I）**：`rollout/__init__.py` **All checks passed**；dev `multi_turn.py` 仅剩**预存 I001**（base HEAD 同样有，非本次引入）；`validate.py` 仅剩**预存 C901**（`_check_async_mode` 经「还原 M12 改动后仍是 13」证明是 Phase 1 的）；twinkle `multi_turn.py` 错误在 L2/167/397（非我改的 `_resolve_call`）。**零新增违规。**
- **ephemeral harness（已删）**：用 **Fake multi_turn engine + MagicMock sampler** 驱动真实 `RolloutEngine` 的 per-episode 机制，验证：非阻塞准入（submit 立即返回、generate 仍 sleep）、arg 透传（adapter_path + allow_partial_rollout 逐字到达 generate、num_samples=1、prompt 包成 [prompt]）、poll before/after done、collect stamping（prompt_id=str(prompt_idx) / policy_version）、collect/poll fail-loudly、collect len!=1（2 与 0 都 raise）、cancel abandon（done 与 in-flight 两种）、close 清理、单轮路径不受影响。

### 7.2 冷审（review-mode）两个真实发现（已修）

1. **GC-drain 是死代码**：初版 `cancel_sample` / `close()` 用 `add_done_callback(lambda f: f.exception())` 去抑制「Future exception was never retrieved」警告。**实测证伪**：Python 3.12.13 的 `concurrent.futures.Future` **根本没有 `__del__`**（探针 `inspect.getsource(Future.__del__)` 报 `AttributeError`）——该警告是 **asyncio.Future 专属**，concurrent.futures.Future 静默丢弃未取用异常。按**剃刀原则移除了这段死代码**，并在 `cancel_sample` docstring 留注释说明 concurrent.futures vs asyncio 的差异，防后人重新加回。
2. **generate C901 复杂度回退**：初版把 `rollout_kwargs` 写成「ternary + 独立 `if allow_partial_rollout:`」使 `generate` 复杂度 10→11（新增 C901）。改用 §5.2 的 ternary-merge 写法（`allow_partial_rollout` 无条件并入、`adapter_path` 保留单一 ternary），复杂度**回到 10**（与 base 等价），C901 消除。

### 7.3 harness 的方法论教训（写正式测试时务必遵守）

初版 harness 有一个 **control 用例失败**（24/25）：我原想用「bare Future set_exception 后 gc 应产生 never-retrieved 警告」来证明 GC-warning 检测有牙，结果 control 测出**无警告**——正是这个失败暴露了 §7.2 发现1（警告机制根本不存在），也说明**没有 control 的反向验证会让「无警告」变成空跑假绿**。教训（呼应 testcase-planning）：每条「不应发生 X」的断言必须配一个「X 确实会被检测到」的 control，否则绿是假的。

---

## 8. 未完成事项（用户明确要求「包括未完成的测试和代码编写」）

### 8.1 代码层面

- **Phase 2 seams 1-3 代码已完整**，无半成品。
- **seam 4（version-span）未写**——按 §4 定案延到 Phase 3，**这是有意为之，不是遗漏**。
- **Phase 2 尚未请用户签收**。接手后**第一件事应是向用户汇报 Phase 2 完成情况并请求签收**，签收后才进 Phase 3（plan 纪律：每 Phase 签收后再进下一 Phase）。

### 8.2 测试层面（正式端到端测试未写）

按纪律，正式测试延后到**用户明确要求**，届时**必须走 `testcase-planning` skill**：先出书面 PLAN 表（case | input | expected | failure it kills）→ 按计划实现 → 每条反向验证（去掉正确行为必须变红）→ 运行结果 + 真实路径覆盖说明。要求：

- **端到端、不允许 mock 孤立手调单方法**（呼应记忆：用真实模板只加载 tokenizer + 脚本化 sampler 返回预置 token + 假 tool_manager，即可在无 GPU 下驱动**真实** `MultiTurnRollout` 跑完整多轮工具循环）。本会话的 ephemeral harness 用的是 **Fake engine + MagicMock sampler**（只验 `RolloutEngine` 的 per-episode 机制），**正式测试要驱动真实 MultiTurnRollout + 真实/脚本化 sampler**。
- 应覆盖（plan L149 的 Phase 2 验证目标）：**多轮 async GRPO 端到端**——episode 粒度 admit/collect、partial rollout 跨轮 abort+resume（in_place 下）、version 锚点（policy_version）正确。
- corner case 从**失败面**推导（逐轮 sample 的 abort/resume 边界、episode 中途失败、group 组装跨 episode、staleness 丢弃整 group、drain 时 in-flight episode 的 cancel、num_generations 条 episode 并发准入的 backpressure 封顶等），每条配 control 防空跑假绿。

---

## 9. Phase 3 详细待办（下一步，签收 Phase 2 后开始）

Plan L151-155「Async 指标」+ 从 Phase 2 延来的 version-span：

1. **StreamingDriver 内部计时器 + 计数器**（twinkle `streaming_driver.py`）：`gen_active_time` / `train_active_time` / `idle_time`；staleness / drop 计数器。通过 seam 或返回值暴露给 dev 侧。
2. **RolloutSample version-span 字段**（dev `rollout/__init__.py`）：新增字段（如 `version_span: int = 0`），由 **driver 端 strategy-aware 标量**填充（§4）：`adapter_snapshot → 0`；`in_place → collect 时 current_version − admit_version`。**引擎（twinkle MultiTurnRollout / sampler）保持版本无关，绝不要把 version 塞进引擎。**
3. **tracking.py metrics 发射**（dev `recipe/tracking.py`）：新增 async metrics 字段 `partial_ratio` / `max_partial_span` / `stale_samples_processed` / `dropped_stale_samples` / `trainer_idle_ratio` / `rollouter_idle_ratio` / version-span 分布，经 `RunTracker.log(metrics, step)` 发射。
4. **验证**：指标非零、与 driver 状态一致。

**Phase 3 注意**：version-span 的填充点在 driver/collect 侧（driver 已知 admit_version 与 collect 时 current_version），不要试图从引擎 trajectory 里捞 version（引擎没有，见 §4）。

---

## 10. Phase 4 / Phase 5 待办（后续）

### Phase 4：sync 路径迁移（staleness=0）

- GRPO/PPO/RFT/GKD 的 `_run_sync` 改为 StreamingDriver（staleness=0）；退役固定批 `generate()`（或保留为 fallback 直到全面验证）。
- RFT：staleness 锁 0（无 IS，不开深缓冲）。GKD：staleness=0 sync / `lmbda==1.0` async。
- **蒸馏族（GKD/OPSD/MOPD）流式迁移**：它们的 one_step_off 仍消费 `overlap_rollout_batches` + `submit_generate`（batch 路径）。**退役 `overlap_rollout_batches` / 移除 `submit_generate` 多轮 guard 的正确时机就在这里**（退役共享函数前必须逐个列出全部消费者并确认本 Phase 无一遗漏——这是 Phase 1 的教训）。
- 届时 `validate.py` 的 M12 蒸馏族拒斥也可放开（蒸馏族有了 per-episode 多轮准入后）。
- **验证**：单轮 sync 行为保持（结果与现固定批一致）；多轮 sync 尾巴消除（GPU 利用率对比）。

### Phase 5：性能实测（写完再处理）

- fully_async 多卡 GPU 全链路真跑；多轮 async 吞吐 / straggler / GPU 利用率；sync streaming vs 旧固定批对比；parameter_sync_step K 调优。

---

## 11. 关键文件索引

| 角色 | 路径 |
|------|------|
| Plan（只读，禁编辑） | `/root/.config/Qoder/.../cache/plans/Per-sample_Streaming_RL_Driver_8c0d73ae.md` |
| StreamingDriver（twinkle 控制面，算法无关） | `twinkle/src/twinkle_agentic/async_rl/streaming_driver.py` |
| 流式 loop mixin（dev） | `swift/dev/recipe/_streaming_loop.py` |
| RolloutEngine（dev 数据面，per-sample/per-episode 接缝） | `swift/dev/rollout/__init__.py` |
| dev 多轮薄包装层 | `swift/dev/rollout/multi_turn.py` |
| twinkle 多轮引擎（数据面，版本无关） | `twinkle/src/twinkle_agentic/rollout/multi_turn.py` |
| twinkle Rollout 基类（`__call__` / `_resolve_call`） | `twinkle/src/twinkle_agentic/rollout/base.py` |
| 配置校验 | `swift/dev/config/validate.py` |
| rollout 配置（parameter_sync_step / async_mode / weight_sync_strategy） | `swift/dev/config/rollout_config.py` |
| async GRPO/PPO loop | `swift/dev/recipe/grpo_async.py`、`swift/dev/recipe/ppo_async.py` |
| sync GRPO/PPO 主 loop（configure_multi_turn 落点） | `swift/dev/recipe/run_grpo.py`、`swift/dev/recipe/run_ppo.py` |
| PromptStream / 旧 batch scheduler | `swift/dev/recipe/train_loop.py` |
| 指标 | `swift/dev/recipe/tracking.py` |
| 核心 sampler partial rollout | `twinkle/src/twinkle/sampler/partial_rollout.py`（`PartialRolloutMixin`） |

---

## 12. 接手步骤（建议顺序）

1. 读本文件 + `.qoder/skills/dev-module-authoring/SKILL.md` + plan 文件（只读）。
2. `git status -s` 与 `git diff --stat HEAD` 确认工作区状态（全部未提交）。
3. 按需回读 §5 列出的 4 个 Phase 2 文件，核实现状与本文件一致（**下结论前先核实源码**）。
4. **向用户汇报 Phase 2 完成情况并请求签收**（不要擅自进 Phase 3）。汇报要结论先行、简练，禁内部编号。
5. 用户签收后：若用户要求补测试 → 走 testcase-planning（§8.2）；若用户要求进 Phase 3 → 按 §9 实施（先出书面 plan/ TodoWrite，AST-only 验证，冷审，ephemeral harness 用完即删）。
6. 全程遵守 §1 环境铁律与 §2.2 分层判据。

---

## 13. 纪律约束（硬规则，勿违反）

- **不编辑 plan 文件**；每 Phase 签收后才进下一 Phase。
- **AST-only 验证**（作者阶段）；正式测试延后到用户明确要求，届时走 testcase-planning（PLAN 表 + 反向验证 + 独立 oracle + 每 RED 反例只留一个违规项 + 断言报错文案归因到目标守卫防空跑假绿）。
- **后端等价不分叉**（vLLM/SGLang 对称，driver 无 backend 分支；sglang 未装只静态核实）。
- **复用既有原语**（RLContextManager / WeightSyncStrategy / PartialRolloutMixin / _train_rollout_batch / _run_micro_step 不动）。
- **不做**：进程分片 / 协程重写 / 动态资源调度 / verl 式 Ray actor 队列（这些是被明确否决的方案，见记忆「多轮 rollout 编排选线程复用引擎而非协程重写」）。
- **剃刀**：不留死代码/半态字段/待开发占位；技术一次性支持到位。
- **fail-loudly**：契约违反必须 raise，不静默退化。
- **ephemeral harness 用完即删**；dev 计划/交接文档（本文件、`RL_*.md`、`*_MIGRATION.md` 等）**仅本地保留，不加入暂存区、不提交**。
- 回复中文 + 标准 LLM 术语（英文技术词保留），结论先行、简练，禁内部编号/代号。
