# dev RL 迁移进度记录（运行日志，随开发持续更新）

> 配套设计文档 `RL_PLAN.md`（工单 W00–W34 与阶段路线的真相源）。本文件只做「防遗忘」的运行清单：
> 当前真实状态、每项的完成度、下一步。进度快照以本文件为准（`RL_PLAN.md` 顶部快照已过期，停在蒸馏 DP=1）。
>
> 工作纪律：写框架代码阶段只做 AST 校验；UT / examples 待用户明确要求再写（届时走 `testcase-planning`：
> 完整 + 端到端 + 反向验证）。多文件编辑串行。剃刀、fail-loudly、只搬不改、通用能力下沉 twinkle。

## 已完成（前序 + 本轮）

- **三新方法 OPSD / MOPD / RFT**：config / cli / recipe / twinkle `MOPDLoss` / examples 全部落地。RFT 过 DP=1+DP=2 真跑冒烟。
- **类层次主干 `TrainLoop`（1b-A）**：`SFTLoop`/`GRPOLoop`/`GKDLoop`/`RFTLoop` 收敛为子类 + 7 个 hook（只搬不改）。
- **`RolloutBatch` 值对象（1b-B）**：列式持有 + 构造期等长校验，消灭平行 list off-by-one。
- **蒸馏对齐 GRPO on-policy 结构（step 1c / W33+W34）**：GKD/OPSD/MOPD 从 `mode='local'` DP=1 就地生成重写为
  Ray DP + 独立采样器 rollout + 逐步权重同步 + design-B mini-batch + 冻结老师 Ray actor（`forward_only`）。
  `plan_rl_device_groups` 扩为 4 参（`teacher_world_size`，向后兼容）；`_TEACHER_GROUP` 独占设备组。
- **HTTP 老师清除（H 类一部分）**：删 `_RemoteGRPOTeacher` + `teacher_tag_key` 死链 + `teacher_model_server` 字段
  + validate.py 7 处守卫；CLI ray-forcing 集合扩到 `{grpo,ppo,rft,gkd,opsd,mopd}`；run_rlhf 分派修复。
- **共享原语归位**：`distill_rows_from_dataset`/`distill_sampling_params` 迁 `_distill.py`，run_mopd 不再依赖 run_gkd。
- 已修真 bug：W00/A1、W02/A2、Bug#3–#9（详见 RL_PLAN §9 阶段一 step 7）。

## 剩余事项（依次推进）

### 一、代码复用收口（阶段一残余 1b，行为等价）
- [x] **1b-C Teacher 协议（W29）** ← 已完成（AST 通过）
      新建中立共享模块 `recipe/_teacher.py`：`Teacher` 结构化 Protocol（唯一契约 `forward_only(inputs,**kwargs)`）
      + 四实现 `DisableAdapterTeacher`（学生 actor `disable_lora=True`，替 `'disable_lora'` 哨兵）/
      `DynamicSelfTeacher`（学生当前权重，消 OPSD 的 `None` 歧义）/`FrozenModelTeacher`（冻结模型，`offload`
      移入其 `__init__`+`forward_only` 包装，暴露 `.model` 供 `_sync_reference`）/`MultiTeacher`（MOPD 的 K 老师
      + 混合权重容器）。共享原语 `response_positions` / `encode_privileged_view`（多模态判据有意统一为
      `is not None`）迁入，`_distill.build_privileged_teacher_feature` 与 `grpo._teacher_feature` 各自保留消息
      构建后委托它。
      **只收敛「谁打分 + disable_lora + offload」，不收敛信号**：GKD=full-vocab `teacher_logits`、
      OPSD/MOPD=response-only `teacher_logps`（MOPD K 通道加权）、GRPO RLSD/SDAR=response-only `teacher_logps`
      是三种不兼容契约，强行合并 `score()` 会改行为、违反「只搬不改」。
      触及：`_teacher.py`(新)/`_distill.py`/`grpo.py`/`run_grpo.py`(`_build_frozen_model` 加 `student` 位参返回
      Teacher 包装)/`run_gkd.py`(删 `offload_teacher_model` loop 参+offload 块)/`run_opsd.py`(删 `_teacher_owner`)/
      `run_mopd.py`(删 `MOPDLoop.__init__`，weights 归 `MultiTeacher`)。
      **范围外（阶段三/四）**：`run_ppo.py`/`run_dpo.py` 各自独立 loop 的 `'disable_lora'` 哨兵不动。
      **连带过时测试（阶段四修）**：`tests/feature/rl/test_recipes.py:153` 仍向 `GKDLoop` 传已删的
      `offload_teacher_model=`（属既有 stale GKD/HTTP-老师 API 测试块）；`:482` 的 `_response_positions`
      已保留为委托 staticmethod，不受影响。
- [x] **1b-D 共享 optimizer-window 步序抽取** ← 已完成（AST 通过，用户选定「最小抽取」）
      核实后挑战 plan：GRPO/RFT/GKD 三个 `fit` 逐字重复的只是 **optimizer-window 步序**
      （`micro_step += 1 → forward_backward → is_grad_sync_boundary → clip_grad_and_step → _record_step`，
      必须与 twinkle grad-sync gate 同相）；GKD 继承 GRPO 属合理复用（仅 2 个中和 override），
      完整三层拆分（OnPolicyLoop/DistillLoop）大多是层级美化而非去重，且阶段四前无测试兜底 → 按剃刀放弃。
      做法：抽 `TrainLoop._run_micro_step(forward_kwargs)` 单一原语（用既有 `_is_grad_sync_boundary()`），
      GRPO/RFT/GKD 的 `fit` 内层改调它（OPSD/MOPD 继承 GKDLoop.fit 自动受益）；SFT 因 micro-step 交织
      callback 事件，保留自己的 cadence 不用它。行为等价（`ga` 每次读 `self.gradient_accumulation_steps`，
      fit 期间不变）。触及：`train_loop.py`(+helper)/`grpo.py`/`run_gkd.py`/`run_rft.py`。
- [x] **1b-E ColocateHandover dedup** ← 已完成（AST 通过）
      核实：`SamplerRollout.sync_weights/finish_generate`（run_grpo）与 `assembly._build_eval_sampler` 的
      `eval_enter/eval_exit` 逐字重复同一 colocate 交接序列（wake weights → sync → offload trainer →
      wake kv_cache；sleep → reload）。**不下沉 twinkle**：`CheckpointEngineManager` docstring 明确「故意
      不替 caller 做内存调度」（只有 caller 知道循环里何时设备空闲；唤醒已醒 sampler 在 vLLM 非 no-op），
      下沉会违背其设计；且两 dev 站点已细化过文档序列（第二次 wake 只请 `kv_cache` 而非 `wake_up()`）。
      做法：新建中立共享 `recipe/_colocate.py` 的 `ColocateHandover(model,sampler,manager,*,colocate,merge_and_sync)`
      （`enter`/`exit`），两站点都改用它（eval 恒 colocate=True；standalone 时 enter 退化为纯同步、exit no-op）。
      删除 SamplerRollout 已死的 `self.colocate`/`self._merge_and_sync`。Bug#8 今后只需改一处。
      触及：`_colocate.py`(新)/`run_grpo.py`/`assembly.py`。

### 二、采样器通用化（阶段二）
- [x] **sglang 接线（W30）+ sglang world size（B3）** ← 已完成（AST 通过）
      `run_grpo.py` 新增三个 backend-aware 共享 helper：`_sampler_backend(rollout_config)`（读
      `rollout_sampler`，非 `('vllm','sglang')` fail-loudly，因 TransformersSampler 无 `CheckpointEngineMixin`
      无法权重同步）、`_sampler_world_size(rollout_config, backend)`（vllm=tp*dp；sglang=tp*pp*dp，修 B3）、
      `_sampler_engine_args(...,backend)` 改为复用 `builders.build_engine_args` 的**全量前缀剥离**（`vllm_*`/
      `sglang_*`→引擎键，取代原来只转 4 旋钮，修 W30「全量映射器闲置」），colocate 时按 backend 强制
      `enable_sleep_mode`(vllm)/`enable_memory_saver`(sglang)。6 个 recipe（grpo/rft/gkd/opsd/mopd/ppo）
      统一改为 `backend=_sampler_backend(...)` → `_sampler_world_size(...,backend)` → `build_sampler(backend=backend)`。
      `validate._check_rollout_sampler` 从「只放行 vllm」改为放行 `('vllm','sglang')`、拒绝其余；
      `rollout_config.rollout_sampler` 注释同步（sglang 已接线）。
      **静态核对**：8 文件 AST 通过；grep 确认无遗留 `backend='vllm'` 硬编码 / 旧 world-size 计算 /
      旧 `_sampler_engine_args(...)` 三参调用；`build_engine_args(backend, None, rollout_config)` 位置参对齐
      （infer_config 仅 transformers 分支读，已被 `_sampler_backend` 排除，传 None 安全）。
- [x] **`vllm_mode='server'` → `disaggregated` 改名/重释（保留 alias）+ CheckpointEngineManager mode 校正** ← 已完成（AST 通过）
      `RolloutConfig.vllm_mode` Literal 从 `['server','colocate']` 扩为 `['disaggregated','server','colocate']`：
      `'disaggregated'` 为 canonical（分离式 Ray 设备组、NCCL 权重同步），`'server'` 降为**历史误名 alias**
      （非 HTTP server），`None` 仍走分离式。加 `#:` 注释澄清（字段名 `vllm_` 前缀亦为历史，现对 vllm/sglang 通用）。
      `plan_rl_device_groups` 逻辑无需改（`if vllm_mode=='colocate' else 分离式` 已覆盖 disaggregated/server/None），
      仅更新其 docstring、模块顶部两种放置说明与 colocate 越界错误信息（`"server"`→`"disaggregated"`），
      并顺手修正模块 docstring 里 stale 的 `CheckpointEngineManager(colocate=False/True)` → `mode='standalone'/'colocate'`。
      **CheckpointEngineManager mode 校正**：核实 W00/A1 已在阶段一修复（`SamplerRollout.__init__` 用
      `mode=('colocate' if colocate else 'standalone')`、`assembly._build_eval_sampler` 用 `mode='colocate'`），本轮无需再改。
      **静态核对**：2 文件 AST 通过；无 validate/CLI choices 限制 `vllm_mode` 值（Literal 即约束）；`cli/rlhf.py:88`
      RL 默认 `'colocate'` 行为不变；examples 里的 `--vllm_mode server` 全属 legacy `examples/train/`（走 legacy CLI，
      且 'server' 仍有效），dev examples 属阶段四。**连带过时测试（阶段四修）**：`test_cli_mapping.py:375`
      断言 `parse_rollout_configs(...)vllm_mode=='server'`，但 `swift/dev/cli/rollout.py` 前序已删（该测试本就 broken）。
- [x] **`SamplerRollout` → `SyncableRollout` 归并 + W14（基类静默跳过权重同步）** ← 已完成（AST 通过）
      **W14 根因**：`grpo.py::_generate` 与 `run_ppo.py::PPOLoop.fit` 都用 `if hasattr(self.rollout,'sync_weights')`
      守卫；基类 `RolloutEngine` 无此方法 → 权重同步被**静默跳过**（behaviour policy 停在 INITIAL 权重、
      非算法正确 on-policy RL）且无 warning，易误接未完成基类。
      **修**：基类 `RolloutEngine` 新增 `sync_weights()`/`finish_generate()` 为 **warn-once no-op**（`_warned_no_sync`
      闸门，`logger.warning` 点名“不同步权重=陈旧策略、仅 pipeline smoke，真实运行应用 SyncableRollout”）；
      两个 loop 去掉 `hasattr` 守卫改为**无条件调用**（每个 rollout 现在都有这两个方法，这就是“归并”）。
      **改名**：`SamplerRollout` → `SyncableRollout`（对齐 plan §3.8 協议表词汇；表达“在 RolloutEngine 上加
      权重同步”），run_grpo 类定义 + 5 recipe import/实例化 + `recipe/__init__.py` 导出 + grpo/_colocate/rollout
      docstring 全部同步（全仓 grep 确认无遗留旧名、无外部消费者、无测试引用）。同步修正 rollout 模块
      docstring 里 stale 的“Backend is vLLM-only by design”（s2a 已接线 sglang）与 `separate-server`→`disaggregated`。
      **transformers fail-loudly**：已在 s2a 的 `_sampler_backend` + `validate._check_rollout_sampler` 覆盖（本项无新工作）。
      **静态核对**：10 文件 AST 通过；`SyncableRollout.__init__` 不调 `super().__init__`（故无 `_warned_no_sync`）
      但其 `sync_weights` 完全覆盖、不读该标志 → 安全；run_infer 用基类但从不调 sync_weights（无误报 warning）；
      RFT/GKD/OPSD/MOPD 继承 `GRPOLoop._generate`、PPO 自有 fit，两处守卫均已去。

### 三、legacy 功能复原 + gap 收口（阶段三）
- [x] **W03/A4 PPO 崩溃 + W26 PPO 语义收敛** ← 已完成（AST 通过）
      **W03 崩溃**：`run_ppo.py::_plan_rollout` 以 4 个位置参调 kw-only 的 `GAEAdvantage.__call__`
      （只允许 `rewards`/`values` 两个位置参，`masks`/`normalize` 均 kw-only）→ 首个 rollout 步必 TypeError，
      PPO 从未跑通。**修**：`gamma`/`lam` 是 GAE **构造**超参，改由 `GAEAdvantage(gamma=rlhf_config.gamma,
      gae_lambda=rlhf_config.lam)` 绑定（顺带接线 W25 的 `gamma`/`lam` 死参），调用改 `self._gae(rewards, values)`；
      GAE 返回 `[1,T]` 张量，`squeeze(0).tolist()` 归一为 per-token float list，与 plan 里 `values`/`rewards`
      表示一致（避免跨 `num_ppo_epochs` 持有 GPU 张量）。
      **W26 值损失死参**：`configure_ppo_value_loss` 传 `cliprange_value=`/`vf_coef=`，但 twinkle
      `PPOValueLoss.__init__(epsilon, ...)` 把两者吞进 `**kwargs` 忽略 → `cliprange_value` 被吞（默认 0.2 巧合相等）、
      `vf_coef` 从不乘进值损失。**下沉 twinkle 修**：`PPOValueLoss.__init__(cliprange_value=0.2, vf_coef=1.0,
      ignore_index=-100, **kwargs)` 用真实 PPO 超参名（对齐 dev config / TRL / 测试契约 test_recipes.py:359-361），
      clip 用 `cliprange_value`、`LossOutput(loss=vf_coef * loss)`。全仓无 `PPOValueLoss(epsilon=)` 调用方，
      clean rename。
      **范围裁定**：`outputs.get('values')` 是**正确**运行期契约（transformers.py:730-732 在 `require_values`
      时把 `logits`→`values`；megatron 同），per-token 测试传 `outputs={'logits'}` 属 W32 测试漂移（阶段四修）。
      W26 另记的「无 KL 早停/自适应 KL/熵奖励」：config 无对应字段（无 `target_kl`/`entropy_coef`），属 E 类
      未实现特性缺口、非死参 → 按剃刀不臆造，诚实记录于此。
      **静态核对**：2 文件 AST 通过；grep 确认无遗留 `self._gae(...)` 位置参调用 / 无 `PPOValueLoss(epsilon=)` /
      value.py 无残留 `.epsilon`；`cfg` 仍被 `whiten_rewards` 使用（未变死变量）。
- [x] **C 类 ~31 死参数接线/fail-loudly（W20/W21/W23/W24/W25 + 散参）** ← 已完成（AST 通过）
      按「接线 or fail-loudly 二选一」逐项核实**最内层真实读取点**（grep swift/dev + twinkle/src，排除 config 声明）后分派：
      **接线（twinkle loss 已支持、dev 从不转发）**：① W20 `tau_pos`/`tau_neg`→`SAPOLoss`（仅 grpo `loss_type='sapo'`）；
      ② W23 `max_completion_length`→`DRGRPOLoss`（仅 `'dr_grpo'`）——为此把解析后的 `grpo_loss_type` 从
      `configure_rlhf_loss` 线程传入 `_rlhf_loss_kwargs`/`_online_loss_kwargs`（变体超参只被各自子类读，传给基类
      GRPOLoss 会落 `**kwargs` 静默丢弃）；③ W21 `rpo_alpha`→`DPOLoss.sft_weight`（核实 dpo.py:304-306 = chosen 上 NLL，
      正是 RPO 项）、`reference_free`→`DPOLoss`（**必须与 run_dpo `_build_reference` reference_free→None 成对**，
      否则无 ref logps 且 reference_free=False 会命中 dpo.py:291-293 **零损失**静默不训练）；dpo/kto 同映射 DPOLoss，落点正确。
      **fail-loudly（零读取点、特性未移植；仅对用户显式设置触发，用既有 `_changed_fields` 交集模式，对齐
      `_check_megatron_runtime_configs`）**：validate.py 新增 `_check_unwired_rlhf_knobs`（12 项：`ld_alpha`/`discopop_tau`/
      `loss_weights`/`f_divergence_type`/`real_tau`/`num_generations_eval`/`num_mini_batches`/`local_rollout_forward_batch_size`/
      `num_sample_generations`/`missing_eos_penalty`/`router_replay_mode`/`offload_bridge`）+ `_check_unwired_rollout_knobs`
      （8 项：`async_generate`/`sleep_level`/`move_model_batches`/`offload_optimizer`/`offload_model`/
      `enable_flattened_weight_sync`/`generation_batch_size`/`steps_per_generation`）；W24 在 run_dpo 构造 loop 前守卫
      （配了 eval split 却从不评估 → NotImplementedError，真正实现留 s4）。
      **已接线/已拒（无需再动）**：`vllm_*`/`sglang_*` 引擎旋钮 s2a 已全量前缀剥离接线；`mcore_ref_model`/`_adapter`
      已在 legacy_coverage `unsupported_checkpoint` CLI 拒绝；`teacher_model_type`/`teacher_deepspeed` 已在 `_distill.py` 读取。
      **故意不在此处理**：`desirable_weight`/`undesirable_weight`（KTO 权重）属 W12 语义修复（s3sem），需先让 twinkle
      `kto_pair` 支持再转发，此处不 fail-loudly 以免与 s3sem 冲突。
      **静态核对**：4 文件 AST 通过；grep 确认 dev tests/examples 零引用这些死参（新增 raise 不打断既有用例）；
      `validate_rollout_config` 由 process.py:131 调用、`_check_rlhf_advanced` 由 validate_configs:87 调用（fail-loudly 会触发）。
- [x] **C 类 ~31 死参数** ← 已完成（s3dead；`generation_batch_size`/`steps_per_generation` 后由 B1 收口）：接线或 fail-loudly（`rpo_alpha`/`ld_alpha`/`discopop_tau`/`loss_weights`/
      `desirable|undesirable_weight`/`async_generate`/`sleep_level`/`offload_optimizer|model`/
      `generation_batch_size`/`steps_per_generation`/`prm*`/大量 `vllm_*`/`sglang_*` 引擎旋钮）。
- [x] **B1 `num_train_epochs` 被忽略** ← 已完成（AST 通过，用户选 B｜忠实 generation-batch 分片）：GRPO/PPO/GKD 原每步重生成整个 prompt 集，只有 `max_steps` 生效。
      **修法（用户选 B｜忠实 generation-batch 分片）**：新增共享 `PromptBatchScheduler`（train_loop.py）
      按 `generation_batch_size` 分片、seeded shuffle、跨 `num_train_epochs` 完整遍历循环（None 默认整集→向后兼容），
      yield 全局 prompt 索引；`rollout_step_budget` 与 assembly 同构推导 `max_steps`（显式 >0 用之，否则由 epochs 推导，
      ≤0 fail-loudly）。GRPO/GKD 的 fit 遍历调度器、`_generate/_dynamic_rollout/_rollout_step` 串入 prompt_indices；
      PPO 保持 global_step=每 rollout（其内部多次 clip_grad_and_step 是既有独立 cadence 问题、非 B1），预算=rollout 数。
      `generation_batch_size` 从 `_UNWIRED_ROLLOUT_KNOBS` 移除（已接线）；`steps_per_generation` 保留 fail-loudly 指向
      `num_iterations`（dev 复用旋钮）。RFT 不在范围（有 `rft_iterations`、独立 override fit）。附带修复：`max_steps`
      默认 -1 时 `max_steps or 1`=-1→loop 0 步（不设 `--max_steps` 当前训不了东西），由 epochs 推导一并解决。
- [x] **D/W28 跨栈 legacy import 下沉** ← 已完成（AST 通过）：`swift.rl_core.advantage.*`、`swift.rlhf_trainers.{gkd_helpers,vllm_client,utils}`、
      `swift.infer_engine.*` → 下沉 twinkle 或迁 dev 通用文件。
- [x] **H 类残留** ← 已完成（AST 通过）：`vllm_server_*`（base_url/host/port/timeout/group_port/pass_dataset）移除；legacy
      `_RemoteGKDTeacher`/`gkd_helpers` HTTP gather/infer/scatter 骨架清理。
- [x] **语义 bug** ← 已完成（AST 通过）：W11 ipo β 缩放、W12 KTO 无 KL/非配对崩、W13 gspo 静默降级 token-level、W27 GKD。
      （W15 GKD prefix 已随 step 1c 的 `response_positions` 强制共享前缀修掉；W01 离线 shift 前序核实为已正确。）
- [x] **辅助模型后端通用化 + Ray 放置（s3aux）** ← 已完成（AST + pyflakes 通过）
      两条绝对原则落地到所有 frozen 辅助模型（ref/teacher/reward/scorer）：新建共享原语
      `builders.frozen_auxiliary_distributed_config(distributed_config, world_size, *, parallel_spec=None,
      deepspeed=None)`——只继承 run 的 `backend`+`bridge_backend`（原则1：辅助模型只 `forward_only`，两后端等价，
      megatron 策略得 megatron 辅助）、强制 `mode='ray'`（原则2：driver 不持 GPU，辅助必须是独立 Ray actor）、
      按 `world_size` 定 `nproc_per_node`，不继承 trainer 的并行布局/deepspeed/fsdp。替换掉此前所有
      `DistributedConfig(mode='local')`（无 backend→静默钉死 transformers、且 ray 下 driver 无设备）的辅助构建点。
      **占卡分档（用户裁定）**：LoRA 形态辅助（`disable_lora` 参考、挂策略 base 的适配器）复用策略已加载权重、
      不是独立副本→与策略/sampler 共卡、不建新组；全参形态（全量微调独立 ref、seq_cls 奖励模型、独立 RLSD/SDAR
      teacher）是独立权重副本→各占一个不相交 DeviceGroup（默认 1 rank、teacher 可选 `teacher_parallel_spec`）。
      `plan_rl_device_groups` 扩 `auxiliary_groups: Sequence[Tuple[str,int]]`（teacher 组降为语法糖，同走「追加不相交组」
      循环）；判定与构建同源（`_frozen_model_needs_group`/`_reference_needs_group` 镜像构建分支，防漂移）。
      触及：`builders/model.py`+`__init__.py`（新原语）/`_distill.py`（`build_frozen_teacher` 加 `distributed_config`）/
      `run_grpo.py`（`_build_frozen_model`/`_build_reward_model_scorers` 加 backend+remote_group；`teacher_group_world_size`
      从 `_distill` 迁入 RL 基座避免层级倒置）/`run_ppo.py`（`_build_reference`/`_build_reward_models`）/`run_rft.py`/
      `run_gkd.py`/`run_opsd.py`/`run_mopd.py`（teacher threading）。critic（PPO value model）是可训练模型、用真实
      `distributed_config`+megatron/moe，与策略共 'model' 组，不属冻结辅助、不改。
- [x] **离线偏好族对齐两原则（s3dpo）** ← 已完成（AST + pyflakes 通过）
      dpo/kto/cpo/orpo/simpo/rm 是 RL：`cli/rlhf.py` 把离线族纳入强制 `mode='ray'`（不设 vLLM sampler，无 rollout）；
      `assembly.initialize_twinkle` 加 `auxiliary_groups` 参数（'model' 组后追加不相交组、`nproc_per_node` 增至总 rank 数，
      无 aux 时向后兼容）；`run_dpo` 的全参 frozen reference 从 `mode='local'` driver 进程模型改为 'ref' 组上的独立 Ray
      actor（`_reference_needs_group` 判定同源、`_load_frozen_reference` 用 `frozen_auxiliary_distributed_config`+`remote_group`，
      LoRA/reference-free 不建组）；改写模块顶部 stale NOTE（原述「reference 恒单进程 local」）。
- [x] **PPO B2：每步单样本→design-B mini-batch（s3ppob）** ← 已完成（AST + pyflakes 通过）
      原 `PPOLoop.fit` 内层对每个样本单独 `forward_backward(inputs=[one])`+`clip_grad_and_step`，DP>1 时 slice_dp 无法
      切分单行→「Batch too small」。改为镜像 GRPO design-B：run_ppo 主体算 `train_batch_size = per_device_train_batch_size
      × dp_size`（`build_ray_dp_mesh(distributed_config).data_world_size`）传入 loop；`_plan_rollout` 的 per-sample plans 经
      新 `_plan_mini_batches`（复用 `grpo.split_mini_batches`）切成 `train_batch_size` 行的整 mini-batch，每 mini-batch 一次
      policy + 一次 critic `forward_backward`（parallel lists：advantages/old_logps/returns/old_values 各一行一样本），
      `ga` 个 mini-batch 一个优化器步。rollout 不足一个 mini-batch 时 fail-loudly（rollout 宽度恒定，属配置错误）；
      PPO 仍「一 rollout 记一步」的既有 cadence 不变。
- [x] **padding_free 对 dpo/kto 正确性复核（s3padfree）** ← 已复核：无 bug，不改代码
      `forward_only` 恒调 `unpack_packed_sequences`（transformers.py:876），把输出归一为**按提交序、每特征右填充**的行
      （packed 时按 `position_ids==0` 边界拆分、非 packed 时 no-op），与 `padding_free` 标志无关。故 `_ref_logps`/
      `_sequence_logps` 的右填充分特征假设在 padding_free 下成立，DPO/KTO 的 chosen/rejected 交织（偶奇索引）因 unpack
      保序而存活。**刻意不给 frozen 辅助传 `padding_free`**（`configure_frozen_adapter` 只 `set_processor(InputProcessor)`）：
      这只是 benign 的性能不对称（辅助走填充路径、数值相同），而强传该标志会触发 megatron `padding_free requires
      variable_seq_lengths=True` 守卫（megatron.py:1908）、反而给冻结 megatron 辅助引入失败。validate `_check_rlhf_padding_free`
      仅放行 grpo/dpo/kto/gkd（镜像 legacy，cpo/orpo/simpo/rm fail-loudly），无需扩。

### 四、收口验收（阶段四，test-gated — 待用户发话）
- [ ] W31/W32 破损测试修复（两份 `test_recipes.py`、`reward_funcs=`→`orm`、PPO 张量测试全 SKIP）。
- [ ] 本轮重写连带的过时测试：`test_recipes.py:195-225`（旧 GKDLoop/HTTP 老师 API）、`test_cli_mapping.py:267`（已删 `teacher_model_server`）。
- [ ] 12 算法逐一 e2e + 数值 oracle；蒸馏 DP=1/DP=2 双复核；GRPO example（`examples/v5/rl/grpo/`）。

### E 类未实现（诚实记录，非本轮目标）
gym/env-scheduler（多轮工具环境的调度器/沙盒编排）。
（原列的「多模态 GRPO、异步生成、MoE 路由重放、RLHF 序列并行、训练内 PRM」已由「六项能力补齐」批次全部落地，见更新日志对应
条目；「Megatron 后端 ref/teacher/reward/critic」已由 s3aux 消解：辅助模型经 `frozen_auxiliary_distributed_config`
继承 run backend，megatron 策略得 megatron 辅助，两后端 `forward_only` 等价，非未实现项。）

## 更新日志
- 建立本文件；核实蒸馏重写后老师抽象的真实散落面，开始 1b-C。
- **1b-C 完成**：新建 `recipe/_teacher.py`（Teacher Protocol + 四实现 + `response_positions`/`encode_privileged_view`），
  收敛 grpo/_distill/run_grpo/run_gkd/run_opsd/run_mopd 的哨兵与 isinstance 判断；7 文件 AST 通过；grep 核实
  `'disable_lora'` 仅剩 run_ppo/run_dpo（范围外）与文档/注释。下一步 1b-D（OnPolicyLoop/DistillLoop 中间基类）。
- **1b-D 完成（最小抽取，用户选定）**：核实后挑战 plan 的三层拆分——真正的重复只是三个 `fit` 里的
  optimizer-window 步序。抽 `TrainLoop._run_micro_step`，GRPO/RFT/GKD 内层改调它（OPSD/MOPD 继承受益）；
  4 文件 AST 通过，无遗留 `is_grad_sync_boundary`。下一步 1b-E（ColocateHandover dedup）。
- **1b-E 完成**：新建 `recipe/_colocate.py` 的 `ColocateHandover`，SamplerRollout 与 assembly 生成式 eval 共用
  同一 colocate 交接序列（不下沉 twinkle，因 CheckpointEngineManager 故意不做内存调度）；3 文件 AST 通过。
  **代码复用收口（1b-C/D/E）全部完成**；下一步进入阶段二（采样器通用化 sglang 接线）。
- **s2a 完成（sglang 接线 W30 + world size B3）**：run_grpo 抽 `_sampler_backend`/`_sampler_world_size`/
  backend-aware `_sampler_engine_args`（复用全量 `build_engine_args` 前缀剥离）；6 recipe 统一读
  `rollout_sampler` 选 backend；validate/rollout_config 放行 sglang、拒绝 transformers。8 文件 AST 通过 +
  grep 静态核对无遗留硬编码。下一步 s2b（`vllm_mode='server'`→`disaggregated` 改名 + CheckpointEngineManager mode 校正）。
- **s2b 完成（vllm_mode 改名 disaggregated）**：`vllm_mode` Literal 加 canonical `'disaggregated'`、`'server'` 降为
  历史 alias（非 HTTP）；run_grpo docstring/错误信息统一用 disaggregated 并修正 stale 的 `CheckpointEngineManager(colocate=)`
  → `mode=`。核实 mode 校正（W00/A1）阶段一已修，无需再改。2 文件 AST 通过。下一步 s2c
  （`SamplerRollout`→`SyncableRollout` 归并 + W14 基类静默跳过权重同步加 warning）。
- **s2c 完成（SyncableRollout 归并 + W14）**：基类 `RolloutEngine` 新增 warn-once no-op 的
  `sync_weights`/`finish_generate`（W14：不再静默跳过同步）；grpo.py/run_ppo.py 去掉 `hasattr` 守卫改无条件调用；
  `SamplerRollout`→`SyncableRollout` 全仓改名（run_grpo + 5 recipe + recipe/__init__ + docstring）。transformers
  fail-loudly 已在 s2a 覆盖。10 文件 AST 通过 + grep 无遗留旧名/旧守卫。**阶段二（采样器通用化）全部完成**；
  下一步进入阶段三 s3ppo（W03/A4 PPO 崩溃：位置参调 kw-only GAE + PPO 语义收敛 W26）。
- **s3ppo 完成（W03/A4 PPO 崩溃 + W26 值损失死参）**：`run_ppo.py` GAE 改构造参数绑 `gamma`/`lam` +
  `self._gae(rewards, values)` kw 调用（W03 TypeError 消除，顺带接线 W25 的 gamma/lam），输出归一为 float list；
  下沉 twinkle `PPOValueLoss` 认 `cliprange_value`/`vf_coef`（替换被 `**kwargs` 吞掉的 `epsilon`）、`vf_coef` 乘进
  值损失（W26）。核实 `outputs['values']` 契约正确、per-token 测试漂移属 W32（阶段四）。KL 早停/自适应/熵奖励
  无 config 字段 → E 类诚实记录不臆造。2 文件 AST 通过 + grep 静态核对无遗留。下一步 s3dead（C 类 ~31 死参数接线/fail-loudly）。
- **s3dead 完成（C 类死参接线/fail-loudly）**：逐项 grep 核实最内层读取点后分派。接线 4 组（sapo `tau_pos/tau_neg`、
  dr_grpo `max_completion_length`、dpo/kto `reference_free`+`rpo_alpha`→`sft_weight`；为此把 `grpo_loss_type` 线程传入
  loss-kwargs 装配，reference_free 与 run_dpo `_build_reference`→None 成对避免零损失陷阱）；fail-loudly 20 项
  （validate.py `_check_unwired_rlhf_knobs` 12 + `_check_unwired_rollout_knobs` 8，用 `_changed_fields` 交集仅对显式设置触发）
  + W24 eval 死线守卫（run_dpo，实现留 s4）。`vllm_*`/`sglang_*` s2a 已接线、`mcore_ref_*` CLI 已拒、`teacher_model_type`/
  `teacher_deepspeed` 已读取 → 无需再动；`desirable/undesirable_weight` 归 W12（s3sem）。4 文件 AST 通过 + grep 确认
  dev tests/examples 零引用（新 raise 不破坏既有用例）。下一步 s3b1（B1 `num_train_epochs` 被忽略）。
- **s3b1 完成（B1 `num_train_epochs` 被忽略，用户选 B｜忠实 generation-batch 分片）**：新增 `train_loop.py` 的
  `PromptBatchScheduler`（seeded shuffle + 按 `generation_batch_size` 切片 + 跨 `num_train_epochs` 循环；None 默认整集→
  向后兼容；yield 全局 prompt 索引）+ 三 helper `prompt_batch_count`/`rollout_step_budget`/`resolve_rollout_max_steps`。
  GRPO/GKD/OPSD/MOPD 的 `fit` 遍历调度器、`_generate/_dynamic_rollout/_rollout_step/_round_rows` 串入 `prompt_indices`；
  PPO `fit` 遍历调度器 + 切片 prompts（保持 global_step=每 rollout 语义，预算用 `prompt_batch_count`）。五 recipe
  （run_grpo/run_ppo/run_gkd/run_opsd/run_mopd）重排：prompts/train_batch_size 上移，`max_steps` 由 `num_train_epochs`
  推导（显式 >0 覆盖、≤0 fail-loudly），镜像 assembly dataloader 预算约定。**附带修复**：`train_config.max_steps` 默认
  **-1**（非 0）→ `-1 or 1`=-1（truthy）→ loop `while global_step < -1` 跑 0 步，即不设 `--max_steps` 时 GRPO/PPO/GKD
  当前根本训不了东西，由 epochs 推导一并解决。`generation_batch_size` 从 `_UNWIRED_ROLLOUT_KNOBS` 移除（已接线）；
  `steps_per_generation` 保留 fail-loudly 指向 `num_iterations`。8 文件 AST 通过 + grep 静态核对（`max_steps or 1` 仅剩
  run_rft 有意保留、无 0 参 `_rollout_step`/`_dynamic_rollout` 遗留、`_round_rows` 调用点均带 `prompt_indices`）。
  **诚实记录的既有独立缺陷（非 B1、未顺手改）**：①PPO cadence——global_step=rollout 但内部多次 clip_grad_and_step，
  LR horizon 与优化器实际步数不匹配（B1 前就存在）；②PPO B2——每 forward_backward 只喂一样本，DP>1 会 Batch too small
  （B2 单独工单）；③GKD 混合 lmbda（<1）下 off-policy 轮用 dataset 滚动窗口、mini-batch 数与 on-policy 不同 → LR horizon
  为近似（调度器耗尽仍界定实际轮数）；④resume 不恢复调度器 prompt 位置（既有代码同样无位置概念）。RFT 不在 B1 范围
  （有 `rft_iterations` epoch 类似物 + 独立 override fit + kept 数依赖 reward 无法预先推导预算，默认 max_steps=-1 时
  fail-loudly 报 0 步）。下一步 s3d（D/W28 跨栈 legacy import 下沉）。
- **s3d 完成（D/W28 跨栈 legacy import 下沉）**：grep 核实 dev 当前跨栈引用面（蒸馏重写 + HTTP 老师清除后已大幅收缩）：
  `swift.rlhf_trainers.{gkd_helpers,vllm_client,utils}` **已全无**（随蒸馏重写/HTTP 老师移除消失）；只剩两处运行期
  `swift.rl_core.advantage` 引用 + 一处 `swift.infer_engine` TYPE_CHECKING 引用。做法：①新建
  `twinkle/advantage/teacher_signal.py`（`compute_teacher_logratio`/`expand_advantage_to_per_token`/`apply_rlsd_reweight` 逐字
  下沉，纯 torch 无 legacy 依赖），grpo.py:521 重指；②新建 `twinkle/loss/sdar.py`（`compute_sdar_loss`+`_sdar_agg_loss` 逐字下沉，
  定位为辅助自由函数而非注册 Loss——caller 把它加到 policy loss 上、返回 (loss, metrics)），configure.py:471 重指。
  **故意不动**：①`rewards/orm.py` 的 `from swift.infer_engine import InferRequest` 已在 `TYPE_CHECKING` 下、仅用于引号型注（
  该文件本就是 `swift.rewards.orm` 的 internalized 副本、明言无 legacy 运行期依赖），无运行期跨栈耦合；②`convert.py`/
  `merge_lora.py`/`legacy_dataloader/factory.py` 的 `swift.megatron.*`/`swift.tuners.*` 属 megatron 后端集成（非 W28 点名的
  legacy RL pipeline），是预期架构；③`rollout/__init__.py:58` 的 `swift.rl_core.data` 仅 docstring 说明「故意不 import」。
  **诚实记录**：下沉采 internalize（twinkle 与 legacy `swift/rl_core/advantage.py` 暂时共存，后者仍服务 legacy 训练路径），
  与 orm.py 同模式；未反向让 legacy 依赖 twinkle（避免动 legacy 路径）。4 文件 AST 通过 + grep 静态核对（dev 无残留运行期
  `swift.rl_core`/`swift.rlhf_trainers` import，新 twinkle 符号仅 grpo.py/configure.py 消费）。下一步 s3h（H 类残留：
  `vllm_server_*` 移除 + legacy `_RemoteGKDTeacher`/`gkd_helpers` HTTP 骨架清理）。
- **s3h 完成（H 类残留：`vllm_server_*` 移除 + HTTP 骨架清理）**：grep 核实后删删。①`rollout_config.py` 移除
  `# === External Server ===` 整节（6 字段 `vllm_server_{base_url,host,port,timeout,group_port,pass_dataset}`）；②
  `builders/sampler.py::build_engine_args` 的 `excluded` 集相应收缩为 `{'engine_kwargs', 'mode'}`（6 个 `server_*` 前缀剥离
  键随字段消失而成死项，一并剔除）。**核实无消费者后才删**：全仓 grep `vllm_server_*` → dev 仅 rollout_config 定义 +
  sampler.py excluded 列（无 recipe/validate/cli 读取）；`swift.rlhf_trainers.*`/`swift.megatron.*` 的同名字段是 legacy
  自有副本（不读 dev config，不动）；`legacy_coverage.py` 无按名分类；`examples/v5/` 零引用（`examples/train/*` 绑 legacy
  args_mixin，不受影响）。删后 CLI `--vllm_server_*` 自然变为 Unrecognized（fail-loudly，符合 H 类“无 HTTP server”前提）。
  **`_RemoteGKDTeacher`/`_RemoteGRPOTeacher`/`gkd_helpers` HTTP 骨架**：前序“HTTP 老师清除”已从 dev 源码删除两个类（
  本转 grep 确认 dev 非测试源码零定义、零 `swift.rlhf_trainers` import）；仅剩 `test_recipes.py:168/193/236/259` 的陈旧引用
  （已删类）→ 归阶段四 test-gated。2 文件 AST 通过 + grep 静态核对（rollout_config 无残留 `vllm_server`、`List`/`field` 仍被
  `tools` 使用无未用 import；sampler.py excluded 仅余 engine_kwargs/mode）。下一步 s3sem（语义 bug：W11 ipo β/W12 KTO KL/
  W13 gspo 降级/W27 GKD）。
- **s3sem 完成（语义 bug W11/W12/W13/W27）**：
  **W11 ipo β**：`twinkle/loss/dpo.py` `_compute_dpo_loss` 的 `margin` 收敛为原始 log-ratio 差
  （`chosen_logratios - rejected_logratios`），ipo 分支按定义 `(margin - 1/(2β))²`（此前误把 β 缩放进 margin，
  使 ipo 变成 `(β·margin - 1/(2β))²`）；sigmoid/hinge 分支各自显式乘 β，语义与 TRL 对齐。
  **W13 gspo 静默降级**：`configure.configure_rlhf_loss` 中 `grpo_loss_type=='gspo'` 且 `importance_sampling_level`
  仍为默认 `'token'` 时提升为 `'sequence'`（GSPO 定义即 sequence-level IS，否则被 `_ConfiguredGRPOLoss` 以 token 级包装
  绕过 `GSPOLoss._compute_log_importance_weights`）；`validate._check_grpo_loss_type` 加 gspo+显式 token 冲突守卫（fail-loudly）。
  **W27 GKD resume 错位**：`run_gkd.py` 的 `round_index` 改由 `enumerate(self._prompt_batches)` 得（scheduler 批位置，
  驱动 lmbda 抛硬币 seed 与数据集滚动窗），而非 `global_step`（优化器步数）——GA>1 时两者错位。
  **W12 KTO（用户选 A：真非配对 KTO）**：数据契约从「配对 chosen/rejected 近似（`kto_pair`）」改为 TRL/legacy KTOTrainer 的
  真非配对形。①`twinkle/loss/dpo.py` 新增 `KTOLoss`（PreferenceLossBase 同族）：按 `label` 拆 desirable/undesirable、
  `1 - sigmoid(β·(logratio - z_KL))` / `1 - sigmoid(β·(z_KL - logratio))`、`desirable_weight`/`undesirable_weight` 加权、
  `z_KL` 为 detached 参考点（None→0）、缺 ref_logps 或缺 label 均 fail-loudly、`num_tokens=0`；`loss/__init__.py` 注册 `'kto'`。
  ②`configure._RLHF_LOSS_NAME['kto']='kto'`（改自 `'dpo'`）、`_preference_loss_kwargs` 拆出 kto 独立分支转发两权重。
  ③`run_dpo.PreferenceLoop`：`calculate_kl` 旋钮（`calculate_KL is not False`）；新增 `_encode_kto_batch`（批内轮转 completion
  造错配 KL 批，镜像 legacy `KTOPreprocessor`，复用 template kto 契约的 `rejected_response`）、`_sequence_logps`（用 `_teacher.response_positions`
  归约 forward logps）、`_kto_kl_term`（两次 no-grad forward 算 `mean(seq_logp_π - seq_logp_ref).clamp(min=0).detach()`）、
  `_kto_step_kwargs`（组装 completion 批 + label/z_kl/ref_logps）；`fit` 加 kto 分支（单序列前向，非 interleave）。
  ④`validate._check_preference_reference`：拒绝 kto+reference_free（KTO 无 reference-free 形，log-ratio 与 z_KL 都需参考）；
  `_UNWIRED_RLHF_KNOBS` 注释更新（两权重已接线，移出死参清单）。5 文件 AST 通过 + 静态核对（`response_positions` 契约、
  `calculate_KL`/`desirable_weight`/`undesirable_weight`/`reference_free` 均为 rlhf_config 真实字段、`forward_only` 返回 `logps`）。
  **诚实记录**：轮转造 KL 批继承 legacy 同款约束（相邻 completion 相同会触发 template 的 `rejected != response` assert）。
  〔订正：本条原记「KTO 走 transformers 单进程 local 路径（与既有 dpo 同）」已被 s3dpo 作废——离线偏好族现恒 `mode='ray'`、
  辅助模型后端无关；见下 s3dpo/s3aux 条目。〕下一步 s3aux（辅助模型 ref/teacher/reward 后端通用化 + Ray 放置）。
- **s3aux 完成（辅助模型后端通用化 + Ray 放置）**：见「剩余事项·三」中同名 [x] 条目的完整描述。核心是共享原语
  `builders.frozen_auxiliary_distributed_config`（继承 backend+bridge_backend、强制 mode='ray'、按 world_size 定 nproc）
  替换所有 `DistributedConfig(mode='local')` 辅助构建点；`plan_rl_device_groups` 加 `auxiliary_groups`；LoRA 共卡/全参异构
  独占组（用户裁定）；`teacher_group_world_size` 从 `_distill` 迁入 `run_grpo`（RL 基座）避免层级倒置。9 文件 AST 通过。
  下一步 s3dpo（离线偏好族对齐两原则）。
- **s3dpo 完成（离线偏好族 ray 化）**：`cli/rlhf.py` 离线族 `{dpo,kto,cpo,orpo,simpo,rm}` 强制 `mode='ray'`（无 sampler）；
  `assembly.initialize_twinkle` 加 `auxiliary_groups`（追加不相交组 + `nproc_per_node` 增至总 rank，向后兼容，既有 SFT/seq_cls/
  embedding/reranker/infer 调用点均用默认值不受影响）；`run_dpo` 全参 reference 改 'ref' 组独立 Ray actor（`_build_reference`/
  `_load_frozen_reference` 加 `distributed_config`+`remote_group`、`_reference_needs_group` 判定同源、改写模块 stale NOTE）。
  核实 ray 模式 `nproc_per_node` 契约与在线族一致（须显式传，缺失 fail-loudly）。3 文件 AST 通过。下一步 s3ppob（PPO B2）。
- **s3ppob 完成（PPO 每步单样本→design-B mini-batch）**：见「剩余事项·三」中同名 [x] 条目。`train_batch_size=per_device×dp_size`
  传入 `PPOLoop`，`_plan_mini_batches`（复用 `grpo.split_mini_batches`）切整 mini-batch，每 mini-batch 一次 policy+critic
  `forward_backward`（parallel lists），修 DP>1「Batch too small」；一 rollout 记一步的 cadence 不变。1 文件 AST 通过。
  下一步 s3padfree（padding_free 对 dpo/kto 正确性复核）。
- **s3padfree 完成（复核，无代码改动）**：`forward_only` 恒 `unpack_packed_sequences` 归一输出为按提交序每特征右填充行，
  与 `padding_free` 标志无关→ref_logps 对齐、interleave 偶奇保序，dpo/kto 正确。刻意不给 frozen 辅助传 `padding_free`
  （benign 性能不对称；强传会触发 megatron `variable_seq_lengths` 守卫）。**s3 阶段（辅助模型 + 离线族 + PPO 批 + padding_free
  复核）全部完成**；剩余为阶段四 test-gated（W31/W32 破损测试、本轮重写连带的过时测试、12 算法 e2e + 数值 oracle），待用户发话。
  全量静态核对：12 改动文件 AST + pyflakes 均通过；`frozen_auxiliary_distributed_config`（def + 7 调用点）、`build_frozen_teacher`
  （+3 wrapper）、`_build_reference`/`_build_reward_models`/`_build_frozen_model`/`_build_reward_model_scorers`、
  `plan_rl_device_groups(auxiliary_groups/teacher_world_size)`、`initialize_twinkle(auxiliary_groups)`、`PPOLoop(train_batch_size)`
  的签名与调用点逐一交叉核对一致。
- **异步生成完成（六项能力·其六，风险最高）**：〔⚠️ 已被取代——本条描述的「driver 侧双缓冲 / `async_generate` 布尔 / 仅 GRPO / staleness 钉死 ≤1 / 拒多轮」是**早期批粒度方案**，后经 per-sample 流式重构（见文末「per-sample 流式 RL driver」条目）整体替换：`async_generate` 降为 `async_mode='one_step_off'` 的 legacy alias，五个在线算法（GRPO/PPO/GKD/OPSD/MOPD）统一走 twinkle 算法无关的 `StreamingDriver` + dev `StreamingLoopMixin`，staleness 成为 `async_mode`(one_step_off=1 / fully_async>1) + `max_staleness` 旋钮，多轮已放开。本条仅作历史留存，勿据此判断现状。〕
      用户拍板决策落地——统一 `GRPOLoop` 上做 driver 侧双缓冲（**不**路由到 twinkle
  原生 `AsyncMultiLoraGRPOPipeline`：那是多 LoRA/TQ/YAML/仅分离式的独立 worker 编排，不含 dev 已接特性），`async_generate`
  布尔开关语义＝**1-batch 前瞻**（staleness 固定 ≤1 个 rollout batch、不设 `max_staleness` 旋钮、与批内重放的 `num_iterations`
  正交）。staleness≤1 是物理约束：权重同步就地改写采样器 live 权重且要求其静止，无法与在飞生成重叠（>1 需 LoRA adapter 版本
  pinning，全参无解，本轮不做）。因 staleness 恒 >0 → 强制搭配 `rollout_importance_sampling_mode` 做 off-policy 校正；仅
  `vllm_mode='disaggregated'`（colocate 单卡分时无法重叠）；仅 GRPO（RFT/PPO 各自 override `fit`，不继承 async dispatch）。
  **① twinkle 下沉**：通用非阻塞生成原语 `GenerationSubmissionMixin`（`submit_generation`/`get_generation_status`/
  `collect_generation`/`cancel_generation`/`cancel_all_generations`，纯 `Future` 记账 + `_dispatch_generation` 允许 prompt 数
  < DP size 的切片）入住 core `twinkle/sampler/generation_submission.py`；`twinkle_agentic/async_rl/generation_submissions.py`
  改为**再导出 shim**（同一 class 对象、单一 MRO 身份，原生管线 import 不破）。core `vLLMSampler`/`SGLangSampler` 混入该 mixin
  + `self._generation_submissions={}` + 把 `sample` 的异步核抽为 `async def _generate_inputs(...)`（DRY：sync `sample` 收敛为
  `_run_in_loop(_generate_inputs(...))` 薄壳，mixin 的 `submit_generation` 复用同一协程，重叠与阻塞路径不可能漂移）；LoRA 用
  async 原语 `_aload_lora`（`await engine._get_or_load_lora`）/`_aregister_lora`（`await asyncio.to_thread(HubOperation.download_model)`
  非阻塞下载 + `await engine.load_lora_adapter`），避免在常驻 loop 内调会 `_run_in_loop` 自死锁的 sync 版。**消重**：删
  `VLLMSamplerTQ._generate_inputs`（与 core 等价，改继承）；`SGLangSamplerTQ` 删冗余 `_aregister_lora`（to_thread 已提升到 core）
  但**保留** `_generate_inputs`（其 logprobs nudge 是 token-in-token-out 正确性所需，非等价）；随之清理两处变未用的 `HubOperation`/
  `Optional` import。
  **② dev rollout**：`GenerationHandle`（携 `submission_id`+`prompt_extras`+`require_logprobs`）+ `submit_generate`（拒 multi_turn、
  守卫 `submit_generation` 可用性、复用 `_build_sampling_params`/`_build_trajectories` 保证与阻塞路径同采样参数）/`collect_generate`/
  `cancel_generate`/`_await_generation`（poll `get_generation_status` 到全 completed 再 `collect_generation`，镜像
  `server.sampler.twinkle_handlers._await_generation` 的 admit/poll/collect 契约，因 collect 未完成会 raise）。
  **③ dev grpo 双缓冲**：`fit` 分派 `_run_async`/`_run_sync`，两者共用 `_train_rollout_batch`（`_plan_mini_batches`×`num_iterations`）
  与 `_assemble_rollout_batch`（打分/优势/PRM/路由重放/IS 重算 old_logps，全在训练/参考/老师模型上、绝不碰采样器）。`_run_async`
  每步：`_collect_generation`（阻塞，采样器转 idle）→ `_submit_generation`（**idle 点**做 `sync_weights` 再非阻塞发行下一批）→
  `_score`+`_train_rollout_batch`（与在飞生成重叠）；权重推送只在 idle 点发生，杜绝在飞竞态，batch_{b+1} 由 v_b 生成、step b+1 由
  v_{b+1} 训练 → staleness 1（由强制 IS 校正）。达 `max_steps` 时 `cancel_generate` 丢弃在飞批而非空等。`run_grpo` 线程
  `async_generate=rollout_config.async_generate`。
  **④ config/validate**：`async_generate` 移出 `_UNWIRED_ROLLOUT_KNOBS` 转活旋钮（补 `#:` 注释说明 1-batch 前瞻语义）；
  `validate_rollout_config` 增 `rlhf_config` 形参、新增 `_check_async_generate`（grpo-only + 强制 IS + `disaggregated` + 拒
  `dynamic_sample`（自适应重生成无法预提交）+ 拒 `multi_turn`）；`process.py` 调用点传入 `configs.get('rlhf_config')`；
  `steps_per_generation` 仍 fail-loudly 指向 `num_iterations`（不变）。
  **review-mode 逮到的关键运行期 bug**：core 采样器本轮新混入 `GenerationSubmissionMixin` 后，两个 TQ 子类仍按旧写法把 mixin
  列在**核心父类之前**（`class VLLMSamplerTQ(GenerationSubmissionMixin, vLLMSampler)`）——mixin 已是核心父类的基，再显式前置会
  重复同一基类、触发 C3 线性化冲突（`TypeError: Cannot create a consistent MRO`），使原生 async_rl 管线**导入即崩**（AST 查不出）。
  修法：基类顺序改为 `(vLLMSampler, GenerationSubmissionMixin)`/`(SGLangSampler, GenerationSubmissionMixin)`（mixin 经核心父类
  传递即得，显式列出仅表意，须置后），并在两处 docstring 钉住该顺序约束防被回改。
  **验证**：11 改动文件 AST 通过；**真实 import** core `vLLMSampler`/`SGLangSampler`（MRO 含单一 mixin、quartet + `_generate_inputs`
  齐备）与两个 TQ 子类（MRO 冲突已消、`_generate_inputs` 覆盖生效）；`_check_async_generate` 7 例守卫矩阵实跑全对（关/非 grpo/
  缺 IS/colocate/dynamic_sample/multi_turn 各自 fail-loudly，干净组合放行）；dev `rollout`/`grpo` import + 方法存在性 + `GRPOLoop.__init__`
  收 `async_generate` + `run_grpo` 线程逐一核实；grep 确认无测试引用 `async_generate` 死参（移除不破既有用例）。**六项能力全部落地。**
- **per-sample 流式 RL driver（取代上一条批粒度双缓冲，当前 async 现状以此为准）**：大任务＝把 dev RL driver 从 **batch 粒度锁步**改为
  **per-sample 流式**——每条 trajectory 单独准入 / rollout / 工具调用，各组件自限并发、互不阻塞；sync/async 不再是两条代码路径，而是
  `async_mode` + `max_staleness` 旋钮上的取值。**控制面在 twinkle**（算法无关 `StreamingDriver`：per-sample 准入 → staleness gate + backpressure
  → as-completed poll → ready buffer → assembly-gated consume → 每 `parameter_sync_step` 步 publish → stale 扫描 → prune → drain，含 per-trajectory
  version pin/release）；**数据面在 dev**（`RolloutEngine.submit_sample/poll_completions/collect_sample`，多轮走 `_submit_episode` 每 episode 一线程）。
  dev 侧算法差异被压到 `StreamingLoopMixin` 的 3 个 hook（`_consume_async_samples`/`_streaming_unit_size`/`_streaming_step_delta`）+ 2 条 assembly
  规则（GRPO 拉完整 group、PPO/GKD 拉单 sample），driver/mixin 内**无任何按算法/模型/sample 类型的分支**。
  **算法覆盖**：五个在线算法全部接流式——`StreamingGRPOLoop`(grpo_async.py)/`StreamingPPOLoop`(ppo_async.py)/`StreamingGKDLoop`(gkd_async.py)/
  `StreamingOPSDLoop`(opsd_async.py)/`StreamingMOPDLoop`(mopd_async.py)；路由在各 `run_*.py` 按 `async_mode` 选 loop（`none`→同步 `_run_sync`；
  `one_step_off`/`fully_async`→ Streaming*Loop）。**RFT 有意只做同步**（有 `rft_iterations` 引导轮 + 独立 `fit`，配 `async_mode` 会 fail-loudly）。
  **staleness 语义**：`one_step_off` 钉 staleness=1；`fully_async` 允许 `max_staleness>=1`（仅 GRPO/PPO）；两者都要求 `vllm_mode='disaggregated'`
  （colocate 单卡分时无法重叠）+ 强制 IS 校正。**publish 机制**：`adapter_snapshot`（每版本存独立 LoRA path、多版本常驻、不打断在飞生成）或
  `in_place`（覆写采样器单一 live 权重，必须配 `allow_partial_rollout`：publish 时 abort 全部在飞、各自 resume 到新权重）。
  **旧批粒度 async 机制已整体删除**：`overlap_rollout_batches`/`submit_generate`/`collect_generate`/`cancel_generate`/`GenerationHandle`/
  `_run_async`/`_submit_generation`/`_collect_generation` 全仓 grep 零命中；base `_drive` 收敛为 sync-only。
  **同步路径未迁上 driver（有意决策）**：`async_mode='none'` 仍走固定批 `_run_sync`——那是唯一支持 colocate 两阶段独占设备交接的路径，
  driver 的单线程后台生成控制流无接缝容纳它；且 staleness=0 下 driver 每次 publish 丢弃全部在飞 v0 样本、无 drain，反不如干净锁步。
  **收尾（本轮）**：① 阻塞生成 bracket（`sync_weights → try: generate finally: finish_generate`）下沉为 rollout 层唯一模块级 `blocking_generate`，
  GRPO `_generate`/GRPO ReMax greedy/PPO `_run_sync` 三处共用（补上 PPO 原内联无 try/finally 的稳固性缺口）；② sampling-params 契约去重——
  `_build_sampling_params` 提为 rollout 模块级函数，单轮阻塞 / 单轮准入 / 多轮 `MultiTurnRollout.generate` 三处共用一份（消除 temperature/old_logps/
  num_samples 的第二份副本漂移风险）；③ 清理全部残留的「1-batch 前瞻 / staleness=0 迁移」过时注释（grpo_async/ppo_async/rollout_config/process/
  _streaming_loop），twinkle 侧 `weight_sync.py` 的 colocate in_place staleness≤1 描述是**框架通用契约、非过时**，保留不动。
  **验证**：改动文件 AST + 真实 import 全过；控制面 ephemeral harness（GRPO/PPO/GKD × adapter_snapshot/in_place + 反向验证）全绿后删除。
  **状态**：代码完成，**正式 e2e/pytest 测试仍按纪律延后到用户明确要求**（届时走 testcase-planning）；性能对标 verl 的实测（Phase 5 性能项）未做。
- **colocate 对标核实（纯读 verl 源码，纠正口径，无代码改动）**：应「verl colocate 复用机制为何比我们好」之问，核实 verl 源码后**推翻先前
  「colocate 真重叠是我们唯一实质差距」的口径**——**verl 的 colocate 与本仓 `ColocateHandover` 是同一类分时独占交接，没有同卡 SM 级
  gen∥train 并发**。verl 每步严格串行 `generate → sleep rollout(释放显存) → train(参数/优化器上卡、结束回落 CPU) → wake + CUDA-IPC
  同步权重 → trainer 回落 → 重建 KV cache`，同步权重时 rollout 处于 sleep，生成与训练时间互斥（`verl/workers/rollout/engine_workers.py:730-830`、
  `trainer_sync.py:35-42`）；`trainer_colocate_async` 名里的 "async" 指 **partial-rollout / 离策略 staleness**，非同卡并发（训练前仍 `abort_replicas()`
  + `sleep_replicas()`）。**真 gen∥train 重叠在两边都只存在于训推分离模式**（verl `separate_async`/`one_step_off_policy`/`fully_async` 全部
  `assert not hybrid_engine`，即独立设备组；本仓 `vllm_mode='disaggregated'` 已覆盖）。故 colocate 侧是**平价而非差距**，正好印证本仓「同步路径
  不迁上流式 driver、colocate 只走 `_run_sync`」与 verl 一致。verl 相对本仓**仅交接工程打磨更细、非重叠优势**：`sleep(level=2)` 直接丢弃 rollout
  权重省一次 D2H（`vllm_async_server.py:1414-1441`）、权重走 CUDA-IPC/POSIX-shm 分桶直传免 NCCL group/CPU 中转（`engine_workers.py:807`、
  `bucketed_weight_transfer.py:102-166`）、`weights`/`kv_cache` 分 tag 释放（`checkpoint_engine/base.py:491-507`）、逐 tensor 惰性权重生成
  （`fsdp/transformer_impl.py:977-1034`）、`BaseEngineCtx` 分阶段 offload（`engine/base.py:300-336`）。**检索未发现** per-microbatch offload /
  权重同步与生成尾部重叠 / H2D 重载与 compute 的 prefetch 双缓冲——即 verl colocate 侧不存在真重叠。**可借鉴项**：若日后要压缩 colocate 交接开销，
  方向是上述工程打磨（尤其 CUDA-IPC 直传 + `sleep(level=2)`），而非追求同卡并发。结论已同步写入 `VERL_COMPARISON.md` §7（表格 + 补注）。

