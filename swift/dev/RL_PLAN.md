# dev RL 迁移与扩展实施计划（plan-gated，未获确认前不写产品代码）

> 本文件是 `feature-dev-plan` 要求的书面计划。配套既有资产盘点见 `swift/dev/RL_MIGRATION.md`（legacy RL 全量参考）。
> 约束：一步一步来（用户驱动，单命令推进）；先 plan 后 code；功能复用优先；通用能力下沉 twinkle；swift 代码尽量复用；按最高等级开源框架设计。
> 验证纪律：写框架代码阶段只做 AST 校验；测试待用户明确要求时再补（届时走 `testcase-planning`：完整 + 端到端 + 反向验证）。
>
> **基本原则（用户明确要求，凌驾全文；下文任何与此冲突的描述一律以本原则为准，且冲突处均为待修缺陷的历史快照）**：
> 1. **后端等价**：训练后端 `megatron` 与 `transformers` **除 `generate`（就地生成）外，其余能力完全一致**。`forward_only`/`forward_backward`/`calculate_loss` 等 `TrainableModel` 统一接口两后端都实现，故训练、打分、老师前向、以及全部辅助模型（ref/teacher/reward/critic）一律**后端无关**——不得因后端不同而分叉逻辑、加 megatron/transformers 特化分支，或把某后端能力写成"未实现"。采样器后端 `sglang` 与 `vllm` 同理：除各自引擎专有旋钮外，rollout 语义、权重同步、设备组规划完全一致。**唯一**的后端差异是 `generate`：只有 `TransformersModel` 有就地实现，`MegatronModel.generate` 按设计 `raise`，因此任何 on-policy 生成都必须走独立采样器（vllm/sglang）+ 逐步权重同步这条两后端皆通的路；就地 `generate` 不作为 RL rollout 路径（详见 W34）。
> 2. **RL 恒 `mode='ray'`，不存在任何 `mode='local'` 的 RL 实现**：所有 RL 训练路径（含蒸馏 gkd/opsd/mopd，以及辅助模型 ref/teacher/reward/critic）只在 `DistributedConfig.mode='ray'` 下运行——driver 是唯一 loop 拥有者，模型/采样器/老师/辅助模型都是各自独立进程、各自 `mpu` 的 Ray actor（故 megatron 策略与 megatron 辅助模型天然不冲突）；`mode!='ray'` 一律 fail-loudly。**全文任何"落在 `mode='local'`""targets `mode='local'`""`mode='local'` DP=1"的描述都是待修缺陷的历史快照，绝不可接受、更不得作为设计依据**。torchrun/`mode='local'` 只保留在 SFT/监督等非 RL 路径。
>
> **重构总原则（全局，用户明确要求，凌驾全文；适用于整个 swift/dev↔twinkle v5 重构，不限本 RL 计划）**：
> - **G1 twinkle 定位**：twinkle 是高度组件化、API 通用化、底层化的组件库（大模型训练/推理通用组件 + agentic 工具）；本次重构就是把它作为 swift 的底层库，同时丰富它的组件；它还包含 server/client 的多租户、多 lora 技术，未来希望把 swift 的 model 与 template 用于其 server/client。
> - **G2 复用优先**：重构重心在算法整合与流程简化；swift 侧使用的大部分技术 twinkle 中都有，应当复用；须关注 twinkle 的使用方法，具体查 twinkle 的 cookbook。
> - **G3 一次性到位**：swift 是开发者常用训练框架，所有技术要求一次性支持到位——不允许临时技术方案、后续待开发事项、"某某不支持"、在代码上打补丁；务必以剃刀原则完成框架开发（不做超需求抽象、不搭半成品路由、不留死参数；确需补丁只用 apply_patch）。
> - **G4 可扩展、无特例**：组件须具备可扩展性；如无允许，不得针对某类特殊组件编写特殊代码（例如某技术仅支持 transformers 不支持 megatron）；流程应尽量简化、完整，具备可扩展性与高度可读性，严禁复杂绕弯子的代码。
> - **G5 迁移完整性**：须保证 swift legacy 与 dev 的 CLI 参数一致性、legacy→dev 功能迁移的完整性，严禁功能迁移不完整（除非用户显式要求去掉的部分）。
>
> **一等目标（用户明确补充）**：本任务不只是新增 MOPD/OPSD/RFT 三种算法，而是对现有 dev RL 代码做**系统梳理与重构**，使**所有 RL 算法（既有 9 种 + 新增 3 种 = 12 种）都准确、可用、代码优良**。"准确"= 算法语义与 ground truth 一致、无死参数、无静默降级；"可用"= 每条命令真能跑通并产出正确 checkpoint；"代码优良"= 分层正确、最大化复用、通用能力下沉 twinkle、无跨业务/跨栈乱引用、剃刀（无超需求抽象/半成品路由）、命名与注释教科书化。
>
> **已拍板决策（用户回复"都是 A"）**：D1-A（`rlhf_type` 新增一等值 `opsd`/`mopd`/`rft`）、D2-A（MOPD 全老师加权，下沉 `MOPDLoss`）、D3-A（分阶段，先做三新方法 + 直接相关 bug）、D4-A（RFT 默认 `best_of_n`）、D5-A（老师 logp 用 sampled-token k3 形式，复用 `OPSDLoss`）。详见 §4。
>
> **架构前提（用户明确纠正，贯穿全设计）**：**训练路径下没有"HTTP server"这种东西**。模型与采样器都走 twinkle 的 Ray 机制——`twinkle.initialize(mode='ray', groups=[DeviceGroup(...)])`，model / sampler 都是 `@remote_class` 起的 Ray actor，按 `remote_group` 落到各自的 DeviceGroup；driver 直接通过 Ray 调 `sampler.sample(...)`，不经 HTTP。因此：
> 1. `rollout_config.vllm_mode='server'` 是**历史误名**，实际语义是"分离式(disaggregated) Ray 设备组"（trainer 与 sampler 占不相交 GPU，NCCL 权重同步），与 `colocate`（共享同一组 GPU，CUDA IPC 权重同步）相对。设计里把它**改名/重释为 `disaggregated`**，不再叫 server。
> 2. legacy 里真正的 HTTP 老师路径（`teacher_model_server` / `gkd_helpers` 的 `TeacherServerConfig`+路由 / `run_gkd._RemoteGKDTeacher` / `run_grpo._RemoteGRPOTeacher`）在 v5 RL 训练路径下**移除或重释为 Ray actor**，不再复用其 HTTP 抓取骨架。MOPD 多老师 = 多个冻结 twinkle model actor，各占独立 DeviceGroup，`forward_only` 打分（见 §3.3）。
> 3. `rollout_config.vllm_server_*`（base_url/host/port/timeout/group_port/pass_dataset）在 RL 训练路径下是**死参数**，按"无 server"前提应**移除**（而非接线），保留则在 `_validate` fail-loudly 点名"训练路径不起 HTTP server，sampler 走 Ray DeviceGroup"。详见 §2.C。
>
> （注：HTTP server 只在 `swift deploy`/`swift infer` 的**服务/离线**路径存在，那条路径不受本前提约束；本前提仅针对 RL **训练**路径。）

---

## 进度快照（供接手；更新于阶段一 step 7 完成后）

**当前位置**：阶段一 step 7（最小 e2e 冒烟）**已完成**；下一步 = **step 1b（类层次抽取，§3.8(6)）**，从 **1b-A** 开始。用户已确认 1b 的 4 点：(a) 5 子步顺序、1b-A 先行；(b) drift 用 hook 保持行为不变、不擅自统一语义；(c) `TrainLoop` 落 `train_loop.py`、模块 helper 原地不动；(d) 等价 oracle = 固定 seed 的 loss 序列 + checkpoint 逐位一致。

**已实现且冒烟通过**：
- 三新方法 **OPSD / MOPD / RFT**（config/cli/recipe/twinkle `MOPDLoss`/examples 全部落地）。OPSD·MOPD **仅** DP=1（`mode='local'` 进程内生成）通过——**这是 W33 放置缺陷、非最终形态：蒸馏违反 §1.4「RL 仅 Ray」硬契约，须扩到 DP>1（见 §2.0 W33）；且「进程内生成」仅 transformers 后端可行——megatron 策略 `generate` 必 raise、无前置守卫（W34）；DP=1「通过」不等于验收完成**；RFT DP=1 + DP=2（Ray+vLLM colocate）通过；GRPO DP=1 + DP=2 通过（LoRA+`beta>0` 走 `disable_lora` 参考打分）。checkpoint 均读回、张量全 finite、`lora_B` 非零证明真训练。
- 已修真实 bug：**W00/A1**（`CheckpointEngineManager(mode=)`）、**Bug#3**（辅助模型裸 hub id 未 snapshot_download）、**Bug#4**（ray 无全局 DeviceMesh）、**Bug#5**（driver 读 ray proxy attr → 改调 method）、**Bug#6/design B**（逐样本喂 `forward_backward` → mini-batch 宽 `per_device×dp_size`，DP>1 `Batch too small` 根因）、**Bug#7**（`forward_only` 惰性句柄 → `lazy_collect=False`，下沉 twinkle）、**Bug#8**（colocate 第二次 `wake_up()` 早退 → 只要 `kv_cache` tag）、**Bug#9**（打分侧逐样本 `forward_only` → 按 `train_batch_size` 分块）。
- 注：design B 是「忠实 mini-batch GRPO」——mini-batch 宽 `per_device_train_batch_size × dp_size`、`ga` 跨 mini-batch 累积、`num_iterations` = 复用同一 rollout 的 epoch 数；twinkle GA 边界「晚一步」规则见 §9 step7 与 `rft.sh` 头注。

**功能层面尚未完成（「12 算法皆准确可用」的缺口，非本轮 1b 范围，接手需知）**：
- **W03 [崩溃]**：PPO 以位置参数调 kw-only GAE（`run_ppo.py:433`）→ **PPO 至今完全跑不起来**（阶段三）。
- **W01 [静默训错]**：离线偏好 `chosen/rejected_labels` 从不 next-token shift → dpo/kto/cpo/orpo/simpo 全在自见分布上训练（阶段一列入但**尚未修**，接手核实）。
- **W13** gspo 静默降级 token-level；**W11** ipo β 缩放错；**W12** KTO 无 KL 项/非配对崩；**W15** GKD prefix 不对称；**W26/W27** PPO/GKD 语义缺陷（阶段三）。
- **W02/A2** GRPO `completion_mask` off-by-one：阶段一「RFT 复用前必修」项，接手核实是否已随 design B 收口。
- **W33 [放置缺陷/不可扩展]**：蒸馏家族（gkd/opsd/mopd）落在 `mode='local'` 单进程 **DP=1**，违反 §1.4「RL 仅 Ray」硬契约，大型训练框架不可接受。设计章节（§1.4/§3.3/§3.5/§3.8）本就要求蒸馏走 Ray/DP、老师是 Ray actor；是实现走了捷径（`cli/rlhf.py:85-88` 不翻转蒸馏、recipe 不算 `dp_size`、`batch_size=per_device` 缺 `×dp_size`、老师建 `mode='local'`），进度快照把 DP=1 当「通过」验收。twinkle `forward_only`/`forward_backward` 是 slice_dp 且为两后端统一的 `TrainableModel` 抽象方法，DP>1 完全可行；但 `generate` **不是**统一接口——仅 `TransformersModel` 有 slice_dp 就地实现，`MegatronModel.generate` 按设计 raise（W34），故 W33 的就地生成 DP>1 仅 transformers 策略成立，megatron 策略蒸馏的生成须走 W34 的独立采样器路。修法与时序详见 §2.0 W33/W34（排在 1b 等价重构之后）。

**后续大方向（除 1b 代码复用外）**：
- **阶段二｜采样器通用化**：`backend='vllm'` 硬编码（W30）→ 可选 vllm/sglang；`plan_rl_device_groups` sglang world size（B3）；`RolloutEngine` 静默跳过权重同步（W14）；`SamplerRollout→SyncableRollout` 归并。
- **阶段三｜死参数 + 数据管线 + 词汇清理**：~31 死参接线或 fail-loudly（W20/W21/W23/W24/W25…）；`num_train_epochs` 被忽略（B1）；跨栈 legacy import 下沉（W28）；移除 HTTP `_Remote*Teacher`/`vllm_server_*`/`teacher_model_server`（H 类）；辅助模型（ref/teacher/reward）后端通用化 + Ray 放置（**已完成**，s3aux/s3dpo，见 §3.5 与 RL_PROGRESS）；PPO 每步单样本→design-B mini-batch（**已完成**，s3ppob，修 DP>1 `Batch too small`）；离线偏好族 ray 化（**已完成**，s3dpo）。
- **阶段四｜收口**：修复既有破损测试（W31/W32）；12 算法逐一 e2e + 数值 oracle；代码质量审查；§2.0 矩阵 12×7 全格收口。GRPO example 也在此阶段补（用户决定「example 和后面的 ut 一起处理」）。
- **阶段五｜原 E 类六项重定性为接线/下沉工单（用户明确要求做，见 §2.E + §3.9）**：多模态、异步生成、MoE 路由重放（R2/R3 + 采样重放）、RLHF 序列并行、训练内 PRM、padding_free+packing 全算法放开。经源码核实**无一是能力缺失**（twinkle 原生 / legacy 待下沉 + dev 接线缺口），已从「诚实记录的未实现能力」移出。仍真未实现的只有外部 rollout server（§2.H 已移除）、gym/env-scheduler。（注：辅助模型 ref/teacher/reward/critic 的 megatron 后端**不属** E 类——按基本原则 1 后端无关，阶段三 s3aux 已透传 `backend` 修掉，见 §3.5。）

**未提交改动（勿擅自 commit）**：`grpo.py`/`run_grpo.py`/`run_rft.py`/`assembly.py`/`ARGUMENTS_MIGRATION.md`/`examples/v5/rl/rft/rft.sh`/`RL_PLAN.md`（本仓）；`twinkle/src/twinkle/{transformers.py:749, megatron.py:430}`（Bug#7 的 `lazy_collect=False`，twinkle 子仓）。

**下一步动作（1b-A，详见 §3.8(6)）**：在 `train_loop.py` 顶部抽 `TrainLoop` 主干（公共 `__init__` 计数器/tracker/save 旋钮 + `_is_grad_sync_boundary`/`_reached_max` + 模板 `_record_step`（含 4 个 hook：`_extra_step_metrics`/`_post_step`/`_should_log`/`_consumed_train_samples`）+ 公共 `save`/`resume` + fit 的 gc 括号），把 `SFTLoop`/`GRPOLoop`/`GKDLoop` 改为其子类、只留 drift hook 覆盖。模块级 helper（`is_grad_sync_boundary`/`num_optimizer_steps`/`start|collect|finish_manual_gc`/`save_training_checkpoint`）原地不动。抽完跑既有 e2e 冒烟须与抽前逐位一致（review-mode 核对「只搬不改」），复核通过再进 1b-B。

---

## 0. 本轮范围与分阶段建议（先决）

用户本轮诉求包含四块，体量差异极大：

0. **梳理与重构现有 dev RL 代码**（一等目标）：对既有 9 种算法逐一审计"准确/可用/代码优良"，修掉真实 bug、接线死参数、收口数据管线缺陷、下沉跨栈 import、复用未用上的 twinkle 能力、修复破损测试。这是贯穿所有阶段的工作流，不是最后补的尾巴。
1. **对比 dev 已有 RL 与 legacy，补齐参数/功能缺口**（§2 的 A–G，其中含数个真实 bug 与大量"声明但未接线"的死参数）——与第 0 块同源，§2 是其缺陷地图。
2. **新增三种训练，全部覆盖在 `swift rl` 命令下**：MOPD、OPSD、拒绝采样微调（RFT）。
3. **通用化**：所有训练方法 × 所有模型后端（transformers / megatron / unsloth / liger）× 所有采样器（vllm / sglang / transformers）× 同卡(colocate)/异构(disaggregated) 放置。

第 3 块是横切重构，牵动 GRPO/PPO/RFT 的 rollout 采样器构造、权重同步、设备组规划；第 0/1 块里"死参数接线"和"数据管线缺陷"（B/C 类）也各自是大工程。若一次性全做，单轮不可验证、风险不可控，且与"一步一步来"冲突。

**排序（D3-A 已拍板，详见 §9）：** 先做一次全量审计（阶段 0，产出每算法缺陷地图，纯读不改），再分阶段实现，每阶段单独 AST 校验 + 让用户验收后再进下一阶段；重构（第 0 块） woven 进每个阶段，最终由阶段 4 逐算法 e2e 收口，确保 12 种算法全部准确、可用、代码优良。

- **阶段 0（审计，纯读不改）**：对既有 9 种算法逐一过 §2.0 审计矩阵，把 §2 的 A–G 缺陷落到"每算法×每维度"的具体清单，作为重构地图。
- **阶段一**：新增 MOPD / OPSD / RFT 三种训练，最大化复用既有原语；顺带修掉与三者直接相关的真实 bug（如 `CheckpointEngineManager(mode=)` TypeError，RFT 复用 GRPO rollout 时会踩到）。
- **阶段二**：rollout 采样器通用化（把 RL rollout 的 `backend='vllm'` 硬编码抽成可选 vllm/sglang，接缝走既有 `build_sampler`/`_derive_sampler_type`），并修 `plan_rl_device_groups`/权重同步在非 vllm 下的路径。
- **阶段三**：死参数接线（C 类）与数据管线缺陷（B 类）逐项收口；跨栈 legacy import 下沉（D 类）；复用未用上的 twinkle 能力（F 类）；辅助模型（ref/teacher/reward）后端通用化。
- **阶段四（收口）**：修复既有破损测试（G 类）+ 每算法逐一 e2e 验收 + 代码质量审查（分层/复用/剃刀/命名/注释），确保 12 种算法全部准确、可用、代码优良。

本文件把四块都设计清楚（保证架构一致、不留半成品路由），但**实现按阶段推进**，每阶段单独 AST 校验 + 让用户验收后再进下一阶段。

---

## 1. Scope & external contract（每种训练"正确"的判据与 ground truth）

### 1.1 OPSD（On-Policy Self-Distillation，在线策略自蒸馏）
- **是什么**：同一个模型既当学生又当老师，二者只在上下文不同——学生只看问题（query-only prompt），老师看"特权上下文"（问题 + 参考答案/诊断/rubric）。学生用自己的当前权重在线采样轨迹，老师在**同一段 response token**上（仅 prompt 不同）给出 per-token 分布，训练最小化学生与老师的 per-token 散度。无需外部老师、无需 reward/advantage。
- **ground truth**：
  - 论文：Zhao et al., *Self-Distilled Reasoner: On-Policy Self-Distillation for LLMs*, arXiv:2601.18734。
  - 既有实现（本仓）：`twinkle/src/twinkle/loss/opsd.py::OPSDLoss(GRPOLoss)`（已注册 `torch_loss_mapping['opsd']`，sampled-token k3 形式：`r = teacher_logp - student_logp`，`per_token = exp(r) - r - 1`，BNPO 式 token-mean 聚合）；legacy 数据侧原语 `swift/rl_core/data.py::OnPolicySample.build_teacher_view()/to_teacher_template_dict()`、`swift/rlhf_trainers/gkd_helpers.py::{encode_teacher_view, build_opsd_samples, remap_teacher_logps_to_student_frame}`；legacy 示例 `examples/train/rlhf/opsd/{opsd.sh, opsd_plugin.py}`（走 `--rlhf_type gkd` + `teacher_prompt` 数据列）。
- **正确判据**：学生 forward 的 response token 与老师 forward 的 response token **完全一致**（同 id、同序），只有 prompt 不同；老师 logp 以 response-only 形式对齐到学生 loss mask（`OPSDLoss` docstring 硬约束：不能用整序列右填充形式）；无老师时退化为 0 loss 但仍过 autograd（DDP/FSDP 不见 unused param）。

### 1.2 MOPD（Multi-Teacher On-Policy Distillation，多老师在线策略蒸馏）
- **是什么**：把多个领域老师融合进一个学生。学生自采样轨迹，多个老师对轨迹给出密集 token 级监督，按权重聚合成一个蒸馏目标。相比单老师 OPSD/GKD，MOPD 降低"曝光差距/整合差距"（学生只在单一老师分布上训练导致的跨领域退化）。
- **ground truth**：
  - 论文：*MOPD: Multi-Teacher On-Policy Distillation*, arXiv:2606.30406（Open-MOPD, bytedtsinghua-sia）。
  - 既有可复用原语（本仓）：**数据/视图侧**——`swift/rl_core/data.py::OnPolicySample.build_teacher_view()`、`gkd_helpers.py::{encode_teacher_view, build_opsd_samples, remap_teacher_logps_to_student_frame}`（这些是纯函数，可复用其"老师视图编码 + response-token 对齐"逻辑）。**多老师打分不走 legacy 的 HTTP 路由**——按"无 server"前提（见文首架构前提），legacy `gkd_helpers.{TeacherServerConfig, parse_teacher_model_server, route_samples_to_teachers, fetch_teacher_parsed_by_routing}` 的 HTTP gather/infer/scatter 骨架**不复用**；MOPD 多老师改为多个冻结 twinkle model actor 各占 DeviceGroup、`forward_only` 打分（见 §3.3）。仅"每老师可带 tag / 加权聚合"的**语义**可借鉴，实现另起。
- **正确判据**：见开放决策 D2——聚合语义（全老师加权 vs 按 tag 路由）一旦选定，token 级 KL 的老师通道数、权重归一、response-token 对齐必须与所选语义一致；多老师各自 `forward_only` 打分后按样本序聚合（同 tokenizer 前提下 response token 逐位对齐）。

### 1.3 拒绝采样微调（RFT / RAFT / ReST，rejection sampling fine-tuning）
- **是什么**：介于 SFT 与 RL 之间的迭代式训练。每轮：对每个 prompt 采样 N 条 completion → 用 reward/verifier 打分 → 按阈值或 best-of-N 过滤保留高质量样本 → 在过滤集上做 SFT（cross-entropy）→ 用更新后的策略进入下一轮。
- **ground truth**：
  - RAFT（Reward rAnked Fine-Tuning）官方实现 `RLHFlow/RAFT`（又称 iterative best-of-n fine-tuning / rejection sampling）；RFT（*Scaling Relationship on Learning Mathematical Reasoning*, Yuan et al.）；ReST（Google, *Reinforced Self-Training*）；RLHF Book 第 9 章 Rejection Sampling。
  - 既有可复用原语（本仓）：rollout = `swift/dev/recipe/run_grpo.py::{SamplerRollout, plan_rl_device_groups, _sampler_engine_args, _grpo_sampling_params, _prompt_rows_from_dataset}`；打分 = `swift/dev/reward.py::{get_reward_funcs, compute_rewards_per_func, weight_rewards, build_frozen_reward_model, build_reward_model_plugins, compute_reward_model_scores}` + `swift/dev/rewards/orm.py`（8 个规则奖励）；SFT = `swift/dev/recipe/run_sft.py`/`train_loop.py` 的 cross_entropy 前向反向。**无新 loss**。
- **正确判据**：过滤规则（阈值 / best-of-N / per-prompt top-k）确定性地选出子集；被选样本的 labels 只在 response 段（prompt 段 -100）、next-token shift 与 SFT 一致；迭代轮之间策略权重同步进 rollout 采样器（复用 GRPO 权重同步）；空过滤集要 fail-loudly 或按配置跳过，不静默产出 0 步。

### 1.4 通用化契约（横切）
- **判据**：任一训练方法在 (backend ∈ {hf/transformers, megatron} × tuner_backend ∈ {peft, unsloth} × liger-fused-loss ∈ {on, off} × rollout_sampler ∈ {vllm, sglang, 进程内 generate} × placement ∈ {colocate, disaggregated}) 的**可行组合**下行为一致；不可行组合必须 fail-loudly 并指向正确做法，不静默降级。**已核实的不可行组合（须在校验期前置报错，不许拖到运行期中途崩）**：①`transformers 采样器做独立进程权重同步`（`TransformersSampler` 无 `CheckpointEngineMixin`，docstring 明确故意不提供）；②`megatron 后端 × 进程内 generate`（`MegatronModel.generate` 按设计 `raise NotImplementedError`，错误信息直接指向独立采样器，见 W34）；③`unsloth × megatron`（unsloth 是 transformers 侧 tuner，`builders/model.py:273` 仅支持 causal_lm）。
- **RL 仅 Ray，不支持 torchrun（用户明确）**：RL 训练路径只在 `DistributedConfig.mode='ray'` 下运行，driver 是唯一 loop 拥有者，模型/采样器/老师都是 Ray actor；`mode!='ray'` fail-loudly。故 RL loop 不带 torchrun 的 rank 守卫分支（详见 §3.8(4)）。torchrun 兼容只保留在 SFT/监督路径。

### 1.5 数据集格式契约（SFT 文本 / SFT 多模态 / embedding / reranker / RL）

本节把 dev 当前的数据格式要求写死为契约，作为 RL/SFT 数据管线的 ground truth（三新方法的数据面必须落在这套契约内，不另造格式）。

**（A0）格式主线决策（用户拍板，覆盖全设计）**

1. **主推格式 = OpenAI 标准 `messages`**，直接采用 twinkle 原生数据类型，不再自造：
   - 原始行 = twinkle `Trajectory`（`data_format/trajectory.py:16`）：`{messages: List[Message], tools, user_data, images/videos/audios, prompt}`；`Message`（`data_format/message.py:65-83`）就是 OpenAI 形状——`role ∈ {system,user,assistant,tool}` + `content`（str 或 content-part 列表）+ `tool_calls`/`tool_call_id`/`reasoning_content`。
   - 编码后 = twinkle `InputFeature`（`data_format/input_feature.py:15-43`）：`input_ids/attention_mask/position_ids/labels/completion_mask/loss_scale/channel/length/routed_experts`（+ 原始 `images/videos`）。
   - **RL/ reward 的额外字段（`solution`/`ground_truth`/`teacher_prompt`/路由 tag 等）走 `Trajectory.user_data`**（`(key, json_string)` 打包对，PyArrow 稳定，见 `pack_user_data`/`user_data_get`），取代 dev 现在的临时 `extra` dict 与 legacy 的 `__#solution` hack。
2. **机制上完全复用 twinkle 的 `sampler` + `Template`**（token-in-token-out）：
   - rollout 后**直接用 `SampledSequence.new_input_feature`** 拿可训练特征（`concat_input_feature` 已填 `labels`+`completion_mask`+assistant message，`sampler/generation.py:141-154`），**不再手搓** dev `rollout/__init__.py` 里的 shift/`completion_mask` 重建。
   - 老师-学生共享 response、多轮拼接、RFT 的 response-only，一律用 **`concat_input_feature(..., appended_as=completion/demonstration/context)`**（provenance 角色决定 labels+completion_mask，`template/base.py:230-268`），取代 dev 的 `replace_assistant_response_with_ids`/`_teacher_feature` 手工对齐。
   - `set_template` / `set_processor(InputProcessor, padding_free=...)` / `encode(add_generation_prompt=True)` 走 twinkle 既有链路（cookbook `rl/grpo/grpo.py:45-47,147-161` 为范本）。
3. **兼容 legacy，但主推新格式**：legacy 原始数据集（`query`/`response`/`instruction` 等别名列、`positive_messages`/`label`）仍能被既有 format_converter 归一到 `messages`/`Trajectory`（见 (A) 别名表），是"入口兼容";内部一律转成 twinkle 原生 `Trajectory`/`InputFeature` 流转。文档与 examples 以新格式为主。
4. **embedding / reranker = 多个 `messages` 拼接 + 可选 `label` 字段**（用户明确）：本质是把 anchor/positive/negative 各自的 `messages` 编码后拼接，附带 float `label`（embedding 相似度）或 1/0 相关性标签（reranker）；沿用 twinkle `InfonceLoss`（embedding，`labels` 1-D mask 标组起点）/ `PointwiseRerankerLoss`·`ListwiseRerankerLoss`（reranker）。见 (D)/(E)。
5. **OPSD/MOPD 的老师用「模型」而非「sampler」**（用户拍板，理由：模型自身 logits 更稳定）：老师 = 冻结 twinkle model actor，`forward_only` 取 response-only per-token logp（sampled-token k3，D5-A），**不采用** cookbook GKD 的"第二个 sampler + `prompt_logprobs=topk`"路（`rl/gkd/gkd_on_policy.py:279-291`）。→ 确认 §3.2/§3.3 的 (a) 方案；cookbook 的 (b) sampler 路记录为"已评估、因 logit 稳定性否决"。

**（A0.1）与 legacy 训练期格式的兼容性 diff（研究结论，逐项标注）**

原始数据集层面：legacy 数据集可不改直接喂 dev（dev 复用同一份 `swift/template/base.py`，别名表/message 键/role/媒体归一/encode 输出键/emb·reranker 行格式/packing 布局均一致）。差异集中在编码内部约定与 RL 样本结构：

| 差异点 | 性质 | 处理 |
|---|---|---|
| **label shift 时机**：legacy 存对齐 labels、loss 时 `torch.roll(-1)`；dev 在 encode 时 shift（`_labels_shifted`，非循环） | token 级监督等价，但**持久化 encoded 行不兼容**（跨树交换缓存/预编码数据集会错位） | 主线改用 twinkle `InputFeature`（其 labels/completion_mask 由 `concat_input_feature` 统一产出），消除 dev 私有 shift 约定 |
| **HF `Image(decode=True)` 列**：legacy 有 `_cast_pil_image` 转 `decode=False`；dev 缺 | **dev 缺口(breaking)**，这类多模态数据集出问题 | 阶段三补 dev 侧等价 cast（入口兼容） |
| 列白名单：legacy `remove_useless_columns` 白名单；dev 无、extra 列原生保留 | dev-only 增强（legacy 超集，`__#solution` hack 不再需要） | 保持；extra 统一走 `user_data` |
| `remove_unused_columns=False`/`disable_auto_column_mapping=True`：dev 硬拒绝 | flag 层面 breaking，默认行为一致 | 保持 fail-loudly |
| 别名/键冲突：legacy drop-both/last-wins；dev first-wins | 边缘歧义行行为不同 | 记录；必要时对齐 legacy 语义 |
| RL 样本：dev `RolloutSample`（flat token ids、精简 reward row）vs legacy `OnPolicySample`（nested、reward row 带 `prompt_id/finish_reason/is_truncated/rollout_infos`） | 对读这些 legacy 键的 reward 函数/plugin **breaking** | 主线用 twinkle `SampledSequence`；reward row 需要的字段经 `user_data` 显式透传，缺失则 fail-loudly |

**（A）标准行结构（所有任务共用）**
- 规范列：`messages` / `system` / `query` / `response` / `images` / `videos` / `audios` / `tools` / `objects`；embedding/reranker 另有 `positive_messages` / `negative_messages` / `label`（见 D/E）。
- `messages` 是 `{'role','content'}` 列表。`preprocessor/base.py:353-354` 限定 `allowed_message_keys={'role','content','loss','loss_scale'}`、`allowed_roles={'system','user','assistant','tool_call','tool_response','tool'}`；超出即报错。
- 别名归一（`format_converter/response.py:35-52`）：`prompt/input/instruction/question/problem → query`；`answer/output/target/solution/text/completion/content → response`。多模态别名（`format_converter/base.py:132` `MEDIA_ALIASES`）：`image → images`、`audio → audios`、`video → videos`。
- `system`/`query`/`response` 三列会被折进 `messages`（response 转成末条 assistant），下游只认 `messages`。

**（B）SFT 纯文本**
- 输入行：`{'messages':[{'role':'user',...},{'role':'assistant',...}]}`（或 `query`+`response`，或 `instruction`/`output` 等别名）。
- `template.encode` → `{'input_ids','labels','attention_mask',...}`。
- **label mask**：`template/base.py:1089-1114` 按 `loss_scale` 逐 token 决定——`loss_scale_list[i]>0.0` 则该段 token 进 labels，否则填 `-100`（不计算 loss）；`encoded['labels'][0]=-100`。prompt 段默认 `loss_scale=0` → 全 `-100`，只有 response 段参与 loss。
- **next-token shift**：`dev/template/template.py:43-47` `_shift_labels_next_token = list(labels[1:])+[-100]`，仅当 `is_training` 且 `task` 不在 `_NO_SHIFT_TASK_TYPES={'embedding','reranker','generative_reranker','seq_cls'}` 时施加；用 `SHIFTED_KEY='_labels_shifted'` 打标记避免重复 shift（twinkle forward 内部不再 shift，见 `transformers.py:709-728`）。RL 复用同一 shift 约定（§3.4 RFT / §3.2 OPSD 的老师-学生 token 对齐都依赖它）。

**（C）SFT 多模态**
- 媒体列 `images`/`videos`/`audios` 经 `cast_mm_data`（`preprocessor/base.py:386-401`）归一为 `[{'bytes':None,'path':image}]`（本地路径）或带 bytes 的 dict。
- 文本里用占位标签 `<image>` / `<video>` / `<audio>`（`template/base.py:56`）标记媒体插入点；content-part 列表（OpenAI 风格 `[{'type':'image',...},{'type':'text',...}]`）由 `StdTemplateInputs.remove_messages_media`（`template_inputs.py:107-135`）抽成媒体列 + 回填占位标签。
- encode 产出的多模态键（以 Qwen2-VL 为例，`qwen.py:379-405`）：`pixel_values` / `image_grid_thw` / `pixel_values_videos` / `video_grid_thw` / `second_per_grid_ts`；Omni 类音频：`input_features` / `feature_attention_mask`。这些键必须在 twinkle 模型边界的放行名单内（见 F），否则被丢弃 → 多模态静默失效。
- **判据**：占位标签数 == 媒体条数（不匹配 fail-loudly）；多模态 RL（GRPO 视觉）已由阶段五 W39 接线落地（见 §2.E/§3.9），不再是 E 类未实现。

**（D）embedding 任务**
- 输入行（`StsbPreprocessor`，`llm.py:1060-1085`）：`{'messages':[{'role':'user','content':sent1}], 'positive_messages':[[{'role':'user','content':sent2}]], 'label':score}`。
- `_embedding_encode`（`base.py:509-546`）→ `anchor_*` / `positive_*` / `negative_*` 键，float 型 `label`，**不做 next-token shift**（在 `_NO_SHIFT_TASK_TYPES`）。
- twinkle `loss/infonce.py` 布局：anchor(1) + positive(1) + negatives(n) 拼接，`labels` 是 1-D mask 标记各组起点；`sentence_transformer_model.py:21` `_ST_FEATURE_KEYS=('input_ids','attention_mask','token_type_ids','position_ids')` 决定进模型的键。默认 loss `'infonce'`。

**（E）reranker 任务**
- 输入行（`MTEBRerankPreprocessor`，`llm.py:1127-1148`）：`{'messages':[{'role':'user','content':query}], 'positive_messages':[[{'role':'assistant','content':doc}],...], 'negative_messages':[...]}`。
- `_reranker_encode`（`base.py:548-585`）把 `chosen.messages + positive.messages` 拼接，`labels` 置 1/0；collator 上限 `MAX_POSITIVE_SAMPLES=1` / `MAX_NEGATIVE_SAMPLES=7`；**不做 shift**。
- `reranker`（`num_labels=1` SeqCls，交叉编码相关性分）走 pooling 前向；`generative_reranker`（如 Qwen3-Reranker）走生成，按 `logit('yes')-logit('no')` 打分。twinkle `loss/reranker.py`：`PointwiseRerankerLoss=BCEWithLogits`、`ListwiseRerankerLoss=按组 CE`。默认 `'pointwise_reranker'`。

**（F）RL 行格式**
- prompt 行经 `_prompt_rows_from_dataset`（`run_grpo.py`）：取 `messages`，剥掉末条 assistant（作为待生成 response），`extras` = 所有非 `messages` 列（reward passthrough：`solution`/`target`/`ground_truth` 等，进 `RolloutSample.extra`，打分时由 `compute_rewards_per_func(columns=...)` 作为 kwargs 传给奖励函数）。
- OPSD/自蒸馏的**特权上下文**走数据列 `teacher_prompt`：`OnPolicySample.build_teacher_view()` 用 `teacher_prompt` 替换（末条/首条，取决于 recipe）user 消息构造老师视图，**response_token_ids 保持不变**（老师-学生共享同一段 response token）；`grpo.py:405-426 _teacher_feature` 断言 response token 数不变。MOPD 多老师同理，每老师可有各自 `teacher_prompt`（或共享学生的 on-policy prompt）。
- `RL_MIGRATION.md §5`（L258-278）已有 `rlhf_type → template-mode → 必需列` 表与 mode map `{'kto':'kto','gkd':'train','ppo':'transformers','grpo':'train'}`（默认 `'rlhf'`）。三新方法补进该表：`opsd`/`mopd` 复用 `gkd` 系（需 `teacher_prompt`），`rft` 复用 `grpo`（需 reward 列）。

**（G）字段删除的两个不同机制（用户强调点，必须分清）**
- **机制 1（dev 预处理，删原始源列）**：`dataset.map(..., remove_columns=self._feature_columns)`（`preprocessor/base.py:176-181`）删除**已被消费的原始源列**（如归一前的 `instruction`/`output`/`image`）；`remove_unused_columns=False` 会**直接 raise ValueError**（`builders/dataset.py:139-175`）。dev dataloader 不做 collation（`_identity_collate`，`builders/dataset.py:283-285`）。
- **机制 2（twinkle 模型边界，删未用的编码特征键）**：用户所说"数据集在最后进入模型后删除无用字段"指的是这一层——`twinkle/processor/base.py:723-757 to_transformers_dict` 用**放行名单**过滤：`_keys=['input_ids','inputs_embeds','attention_mask','position_ids','labels','completion_mask','loss_scale','channel','cu_seq_lens_q','cu_seq_lens_k','cu_seqlens_q','cu_seqlens_kv','max_length_q','max_length_k','packed_seq_params','routed_experts','mm_token_type_ids','second_per_grid_ts'] + VLM_CONCAT_FIELDS`，`for key in list(_input.keys()): if key not in _keys: continue`；`VLM_CONCAT_FIELDS={'pixel_values','image_grid_thw','pixel_values_videos','video_grid_thw','input_features','input_features_mask','feature_attention_mask','grid_thws'}`。`transformers.py:670-674` 在 `self.model(**inputs)` 前 pop 掉 `labels`/`loss_scale`/`channel`/`completion_mask`。
- **对本设计的含义**：预处理阶段**不需要**（也不应）再做一次 `remove_unused_columns`——未用的编码特征键由 twinkle 在模型边界按放行名单自动丢弃。三新方法产出的训练特征只要键落在放行名单内即可；新增的多模态/RL 特征键若要进模型，必须先确认在 `to_transformers_dict` 名单里（否则静默丢弃 → 失效），这是一个必须静态核对的运行期契约。

---

## 2. 现状与缺口分析（dev RL vs legacy）

dev RL 已有：`run_rlhf.py` 分派 9 种 `rlhf_type`（offline {dpo,kto,cpo,orpo,simpo,rm}→`run_dpo`；grpo→`run_grpo`；ppo→`run_ppo`；gkd→`run_gkd`）；`advantage.py`/`reward.py`/`rewards/orm.py`/`loss/configure.py`/`rollout/*` 齐备；`rlhf_config.py`(118 字段) + `rollout_config.py`(61 字段) 已声明 legacy 绝大多数旋钮。缺口按性质分类：

### 2.0 每算法审计矩阵（阶段 0 产出，重构地图）

阶段 0 对下列 12 种算法逐一过 7 个维度，把 §2 的 A–G 缺陷落到"每算法×每维度"的具体格子（✓=已准确可用、✗=有缺陷需修、—=不适用）。此矩阵是"所有 RL 算法准确可用、代码优良"的验收底账，阶段 4 逐格收口。

维度定义：
- **参数接线**：该算法声明的旋钮是否都真实生效（无 C 类死参数），无效的是否 fail-loudly/warning。
- **数据管线**：epoch/step 语义、batch 组装、label shift、过滤/重采样是否正确（B 类）。
- **loss/算法语义**：loss 选型与 ground truth 一致，advantage/KL/clip 等数值正确。
- **后端**：策略模型 transformers/megatron + tuner peft/unsloth + liger 是否可行组合皆通、不可行 fail-loudly。
- **采样器/rollout**：在线算法的 rollout 采样器（vllm/sglang/进程内 generate）是否正确、权重同步是否真发生。
- **放置**：colocate/disaggregated 设备组规划与权重同步 mode 是否正确（含 A1）。
- **代码质量/测试**：分层正确、复用充分、无跨业务/跨栈乱引用、无半成品；既有测试是否绿（G 类）。

**审计已完成（阶段 0 产出）。** 下表每格为 ✓（准确可用）/ ✗（有缺陷，附一句 + file:line）/ —（不适用）。缺陷编号 Wxx 对应下方"重构工单清单"。

| 算法 | 参数接线 | 数据管线 | loss/语义 | 后端 | 采样器/rollout | 放置 | 代码质量/测试 |
|---|---|---|---|---|---|---|---|
| dpo | ✗ W21（`rpo_alpha`/`ld_alpha`/`discopop_tau`/`reference_free` 死参） | ✗ **W01**（`chosen/rejected_labels` 从不 next-token shift，5 算法静默训错，`run_dpo.py:296-307`+`transformers.py:709-728`）；✗ W24（eval 死线，传 `eval_*` 但 `fit()` 从不评估） | ✗ W11（`ipo` β 缩放错，twinkle `loss/dpo.py:179/195`） | ✗ W22（无 backend fail-loudly；`padding_free` 对 dpo 静默错） | — | — | ✗ W31（e2e 仅接线、无数值 oracle） |
| kto | ✗ W21（`desirable_weight`/`undesirable_weight` 死参） | ✗ **W01**（同上 shift）；✗ W12（非配对样本崩，`run_dpo.py` pair 假设） | ✗ W12（`kto_pair` 无 KL 项、无 desirable/undesirable 权重） | ✗ W22（`padding_free` 对 kto 静默错） | — | — | ✗ W31 |
| cpo | ✓（`cpo_alpha`→`bc_coef` 已接线，`configure.py:621`） | ✗ **W01**（同上 shift） | ✓（JSD/BC 数学正确） | ✗ W22（无 backend fail-loudly） | — | — | ✗ W31 |
| orpo | ✓ | ✗ **W01**（同上 shift） | ✓ | ✗ W22 | — | — | ✗ W31 |
| simpo | ✓（`simpo_gamma` 已接线，`configure.py:618`） | ✗ **W01**（同上 shift） | ✓ | ✗ W22 | — | — | ✗ W31 |
| rm | ✓（`center_rewards_coefficient` 已接线，`configure.py:627`） | ✓（seq_cls，无 shift 需求，在 `_NO_SHIFT_TASK_TYPES`） | ✓ | ✗ W22（无 backend fail-loudly） | — | — | ✓ W31（`feature/test_recipes.py` 有 5 步真优化器 e2e） |
| grpo | ✗ W20（`sapo tau_pos/tau_neg` 死参且默认值≠twinkle；~30 `vllm_*`/`async`/`offload` 死参）；✗ W23（`dr_grpo` 的 `max_completion_length` 未转发，twinkle `loss/grpo.py:587` 默认 1024） | ✗ W10（B1 `num_train_epochs` 被忽略，只 `max_steps` 生效）；✗ W10（B2 一次 `forward_backward` 只喂一样本）；✗ **W02**（`completion_mask` off-by-one，`rollout/__init__.py:164`，即 task #18） | ✗ W13（`gspo` 静默降级为 token-level，`configure.py:255-269/294/379` 绕过 `GSPOLoss._compute_log_importance_weights`） | ✓（transformers 策略路径可跑） | ✗ W30（`backend='vllm'` 硬编码，`run_grpo.py:261`；`build_engine_args` 全量映射器闲置，`_sampler_engine_args` 只转 4 个旋钮）；✗ W14（基类 `RolloutEngine` 静默跳过权重同步、无 warning） | ✗ **W00/A1**（`run_grpo.py:159` `CheckpointEngineManager(colocate=)` 必 TypeError，签名是 `mode=`） | ✗ W31（两份 `test_recipes.py` 并存；`rl/test_recipes.py:711` 用已改名字段 `reward_funcs=` 必 TypeError；`run_grpo` 零 e2e） |
| ppo | ✗ W25（`gamma`/`lam`/`cliprange_value`/`vf_coef`/`num_mini_batches`/`local_rollout_forward_batch_size`/`num_sample_generations`/`missing_eos_penalty` 全死参） | ✓（labels 移位约定下 `old_logps`/`ref_logps`/`values` 三者对齐正确，无 DPO 式 off-by-one） | ✗ **W03**（`run_ppo.py:433` 以位置参数调 `self._gae(rewards,values,gamma,lam)`，`GAEAdvantage.__call__` 是 kw-only → 首个 rollout 步必 TypeError，PPO 完全跑不起来）；✗ W26（无 KL 早停/自适应 KL/熵奖励；`vf_coef` 未乘进值损失） | ✗ W22（无 backend fail-loudly） | ✗ W30（`backend='vllm'` 硬编码） | ✗ **W00/A1**（共享 `run_grpo` 放置代码，同崩） | ✗ W32（张量测试与实现漂移且本环境全 SKIP `No module named twinkle`；无 e2e；checkpoint 测试用 `object.__new__` 绕过构造函数） |
| gkd | ✓（无死参；`require_logps=True` 属多余计算，`configure.py:196`） | ✗ W27（`fit` 用 `global_step` 起 micro-step 取样，GA>1 时 resume 后 batch/seed 错位，`run_gkd.py:463`）；✗ W15（prefix 不匹配时 student 取尾/teacher 取头不对称，`loss/gkd.py:87-90`） | ✗ W16（采样温度与蒸馏温度复用同一 `temperature`，改采样锐度静默改散度，`run_gkd.py:254`/`configure.py:586`） | ✗ **W34**（megatron 策略 `generate` 必 `raise NotImplementedError`、无前置守卫，首个 rollout 步即崩） | ✗ **W34**（进程内 `model.generate` **仅 transformers**、非后端无关；megatron 须走独立采样器路） | ✗ **W33**（`mode='local'` 单进程 DP=1，违反 §1.4 RL 仅 Ray，须扩 DP>1） | ✗ W28（跨栈 import legacy `swift.rlhf_trainers.*`/`swift.infer_engine.*`，`run_gkd.py:195/200/223/224/429/430`）；✗ W29（老师抽象非多态：`{str 哨兵, model actor, _RemoteGKDTeacher}` 联合类型 + 散落 `isinstance` 派发；多"老师"是按标签路由非加权聚合 → MOPD 无法直接复用）；缺 `fit()` e2e |
| **opsd（新）** | 设计见 §3.2 | 复用 gkd（须先修 W01 shift 语义 / 用 twinkle `concat_input_feature` 免手搓） | 复用 `OPSDLoss`（已注册，k3 sampled-token） | 同 gkd | ✗ W34（进程内 generate 仅 transformers） | ✗ **W33**（同 gkd，DP=1 须扩 DP>1）；✗ **W34**（megatron 生成须采样器路） | 新增 |
| **mopd（新）** | 设计见 §3.3 | 复用 gkd | 下沉 `MOPDLoss(OPSDLoss)`（须先解 W29：老师抽象升级为多态 + 加权聚合） | 同 gkd | ✗ W34（进程内 generate 仅 transformers） | ✗ **W33**（同 gkd；老师须 Ray actor 各占 DeviceGroup，§3.3）；✗ **W34**（megatron 生成须采样器路） | 新增 |
| **rft（新）** | 设计见 §3.4 | 复用 grpo rollout（继承 W02 completion_mask，须先修） | 复用 cross_entropy | 同 grpo | 复用 `SamplerRollout`（继承 W30 硬编码 vllm） | 复用（**W00/A1 必修**否则崩） | 新增 |

#### 重构工单清单（按严重度排序；阶段归属见 §9）

**致命 / 崩溃级（跑不起来或静默训错，最高优先）**
- **W00（=A1）[崩溃]** `run_grpo.py:159` `CheckpointEngineManager(..., colocate=colocate)` → 签名是 `mode=`，必 TypeError；且 `mode` 从不传 → colocate 场景错落到 standalone（NCCL）。GRPO/PPO/RFT 全踩。**修**：`mode=('colocate' if colocate else 'standalone')`。〔阶段一〕
- **W01 [静默训错]** 离线偏好路径 `chosen_labels`/`rejected_labels` 从不做 next-token shift（dev template 只 shift 裸 `labels` 键，`dev/template/template.py:78-84`），twinkle forward 用未移位 labels 做 `selective_log_softmax`（`transformers.py:709-728`）→ dpo/kto/cpo/orpo/simpo 全部在 P(x_t|x_≤t)（自见）上训练。**修**：对 chosen/rejected 的 labels + loss_scale 施加 `_shift_labels_next_token`。〔阶段一，与三新方法同批修数据面〕
- **W03 [崩溃]** PPO `run_ppo.py:433` 位置参数调 GAE（kw-only 签名）→ 首个 rollout 步必 TypeError，PPO 从未跑通。**修**：改 kw 调用 + 把 `gamma`/`lam` 经构造参数传入 `GAEAdvantage(gamma=, gae_lambda=)`（见 W25）。〔阶段三，PPO 收敛批〕

**严重（算法语义错 / 静默降级）**
- **W02（=task#18）** GRPO `rollout/__init__.py:164` `completion_mask` 顺序 off-by-one。**修**：按 `rollout/multi_turn.py:40-60` 记录的约定对齐。〔阶段一，RFT 复用前必修〕
- **W11** `ipo` loss β 缩放错（twinkle `loss/dpo.py:179/195`）。〔阶段三〕
- **W12** KTO 无 KL 项、无 desirable/undesirable 权重；非配对样本崩。〔阶段三〕
- **W13** `gspo` 静默降级为 token-level（`configure.py:255-269/294/379` 绕过 `GSPOLoss._compute_log_importance_weights`）。〔阶段三〕
- **W15** GKD prefix 不匹配时 student 取尾/teacher 取头不对称（`loss/gkd.py:87-90`）。〔阶段三〕
- **W16** GKD 采样温度与蒸馏温度复用同一 `temperature`。〔阶段一，随 OPSD/MOPD 温度字段拆分一起做〕
- **W23** `dr_grpo` 的 `max_completion_length` 未转发（twinkle `loss/grpo.py:587` 默认 1024）。〔阶段三〕
- **W26** PPO 无 KL 早停/自适应 KL/熵奖励；`vf_coef` 未乘进值损失、`cliprange_value` 被吞（twinkle `loss/value.py:19-22` 只认 `epsilon`）。〔阶段三〕
- **W27** GKD resume 后 batch/seed 错位（`global_step` vs micro-step，GA>1）。〔阶段三/四，随 loop 收敛修〕
- **W33 [放置缺陷/不可扩展，违反 §1.4 硬契约]** 蒸馏家族（gkd/opsd/mopd）落在 `mode='local'` 单进程 **DP=1**，大型训练框架不可接受。**根因（实现走捷径，与设计章节自相矛盾）**：`cli/rlhf.py:85-88` 只把 `{grpo,ppo,rft}` 翻转 `mode='ray'`、注释把蒸馏标「不可翻转」；recipe 从不算 `dp_size`/不建 DP mesh；`batch_size=per_device_train_batch_size`（缺 `×dp_size`）；老师用 `DistributedConfig(mode='local')` 构建。而 §1.4/§3.8(4)（所有 RL=`mode='ray'`、老师是 Ray actor、`mode!='ray'` fail-loudly）、§3.3（MOPD 老师 `build_model(..., mode='ray', remote_group=...)`）、§3.5、§3.8(2)（`FrozenModelTeacher`=冻结 model actor）、§5 不变量3、§6 测试5 **全部要求蒸馏走 Ray/DP**。**DP>1 完全可行（已核实）**：twinkle `TransformersModel.generate`（`model/transformers/transformers.py:888`）本身是 `@remote_function(dispatch='slice_dp', collect=collect_dp_and_flatten, lazy_collect=False)`——把 prompts 切给各 data rank、每 rank 用**自己常驻权重就地生成**自己的分片、按输入序收集（docstring：「splits inputs across data ranks, each worker only generates for its own shard」）；`forward_only`/`forward_backward` 同为 slice_dp。故「就地生成」天然 DP>1、无需独立采样器/权重同步，「targets mode='local'」是捷径非技术必需。**修法（= design B 同构应用到蒸馏 + 老师归位 Ray actor）**：(1) CLI 把 gkd/opsd/mopd 纳入 `mode='ray'` 翻转，但**不**设 `use_vllm`/`vllm_mode`（就地生成无独立采样器，区别于 grpo/ppo/rft 的 vLLM colocate）；(2) recipe 算 `dp_size = build_ray_dp_mesh(distributed_config).data_world_size`、`batch_size = per_device_train_batch_size × dp_size`（全局批，镜像 `run_grpo.py:316`/`dataset.py:325`），`generate`/`forward_only`/`forward_backward` 一律喂全局批由 slice_dp 切分；(3) 冻结老师按 §3.3 建 Ray actor（`mode='ray'`+`remote_group`，`plan_rl_device_groups` 扩为 trainer+N teacher 组）；`disable_lora`/dynamic-self 老师在 DP 学生上 `forward_only`（全局批切片）。**与 1b 时序**：1b 是「只搬不改」等价重构（DP=1 基线即 oracle），W33 是行为变更，**排在 1b 之后**在干净结构上施加，不混进等价门禁。**已知扩展性局限（诚实记录）**：GKD 整词表 `teacher_logits` 经 `forward_only(return_logits=True)` collect 回 driver 再 slice 进 `forward_backward`，driver 瞬时持有 `[全局批, seq, vocab]`，DP 越大 driver 显存越紧；OPSD/MOPD 用 response-only sampled-token logps（小张量）无此问题。GKD 大 DP 缓解（top-k 远程老师 / teacher-student 同 rank 前向免 driver 往返）记为后续优化。**〔用户拍板：W33+W34 合并为一次重写〕按已定生成策略，蒸馏 rollout 一律走独立采样器（不再就地生成），故 W33 原设想的「就地 generate slice_dp 撑 DP>1」对**所有后端**都不再适用——蒸馏的 DP 来自采样器自身 DP + `forward_backward`/`forward_only` 的 slice_dp，与 GRPO **完全同构**。因此 W33（放置/DP）与 W34（生成后端通用）**合并为一次重写、一次测完**：把蒸馏对齐 GRPO 的 on-policy 结构（`mode='ray'` Ray DP + **采样器 rollout（设 `use_vllm`/`vllm_mode`，与 grpo/ppo/rft 同）** + 冻结老师 actor `forward_only` + 蒸馏 loss）。分两次做会先建一套 W34 立即丢弃的就地-DP 脚手架（W33 修法(1)「不设 use_vllm」、(2)「model.generate slice_dp」均随本决策作废，以采样器路为准）。〔阶段一，1b 之后〕

- **W34 [后端通用性缺陷，违反 §1.4 硬契约]** 蒸馏家族（gkd/opsd/mopd）的学生生成硬接到就地 `TransformersModel.generate`（`run_gkd.py:402`），而 **`generate` 不是 `TrainableModel` 统一接口**——基类只抽象 `forward_only`/`forward_backward`/`calculate_loss`（两后端皆实现），`generate` 仅 `TransformersModel` 有就地实现。`MegatronModel.generate`/`generate_stream` 按设计 `raise NotImplementedError`（`model/megatron/megatron.py:450-472`：megatron 权重按 TP/PP 分片、用 megatron 参数名，无推理引擎能读；错误信息直接指向「用独立采样器 vLLM/SGLang/Transformers 加载自己那份权重」）。而 `DistributedConfig.backend` 接受 `'megatron'`、蒸馏策略经同一 `assembly.build_model()`（`run_gkd.py:107`）构建 → **megatron 策略蒸馏配置可达、并在首个 `self.model.generate` 处崩，且无前置守卫**（`cli/rlhf.py` 仅 `--seq_kd` 一处 raise）。这与 §1.4「不可行组合必须前置 fail-loudly」、§3.5 原文「transformers 或 megatron 皆可」、复核矩阵「进程内 generate 后端无关 / megatron 待核」直接矛盾（三处已随本工单修正）。**修法**：(1) **前置 fail-loudly**——`backend='megatron' × rollout_sampler='进程内 generate'` 在 `validate.py`（`_check_rollout_sampler`/`_check_distillation`）即 raise 指向 vllm/sglang，不许拖到运行期中途崩（§1.4 不可行组合②，与 W22「离线偏好无 backend fail-loudly」同类）；(2) **后端通用生成路**——蒸馏学生生成改为可走与 GRPO/RFT 同一套 `Rollout` 协议（独立采样器 + 逐步权重同步；`TransformersModel`/`MegatronModel` 都混入 `CheckpointEngineMixin`，采样器路两后端皆通，复用 design B 的 `SamplerRollout`/`SyncableRollout`/`plan_rl_device_groups`，剃刀不另造）；**就地 generate 不再用于 rollout，只保留给训练中间 eval（transformers-only，见下「用户拍板」③）**；(3) **reconcile §3.8**——`DistillLoop` 挂在 `OnPolicyLoop` 下、而 `OnPolicyLoop` 持 `Rollout`+权重同步 cadence（§3.8(1)/(2)），故蒸馏学生生成本就该走 `Rollout` 协议；§3.5 原文「不用独立采样器」与 §3.8 的 `Rollout` 抽象自相矛盾，以 `Rollout` 协议为准（就地 generate 不再作 rollout、只留给训练中间 eval，见下「用户拍板」③）。**〔用户拍板 2026-10，按用途分派生成机制，三问据此收口〕** ①**学生 on-policy 生成（rollout）一律走 vllm/sglang 独立采样器 + 逐步权重同步**（后端通用；**不再保留就地 generate 作 rollout 的 transformers fast-path——rollout 只此一条路**，最统一，且让 `DistillLoop` 真正落进 `OnPolicyLoop` 的 `Rollout` 抽象、消除 §3.5↔§3.8 矛盾）；②**老师打分是 `forward_only`（就一次前向），继续走冻结 model actor**（`forward_only` 是统一接口、两后端皆通，无需采样器）；③**就地 `model.generate` 只保留给「训练中间 eval 生成」，且仅 transformers**（megatron 无就地生成 → 无采样器场景的中间 eval 生成不支持；但蒸馏/GRPO/RFT 因 rollout 已常驻采样器，其中间 eval 生成应**复用该采样器**，故 megatron 也能 eval——就地 eval 路只是「无采样器训练（如 SFT）」的 transformers 便利）。**取舍（诚实记录）**：rollout 走采样器要为蒸馏付「第二份权重 + 每步学生→采样器同步」（与 GRPO 同代价），连单卡 transformers 蒸馏也要起 vLLM colocate（GRPO 已验证可行）；换来生成路唯一、后端通用。**与 W33 关系**：W33 解决放置/DP（local→ray、batch×dp_size、老师 Ray actor），W34 解决生成的后端通用性；megatron 策略蒸馏的 DP>1 依赖 W34 的采样器路（就地路在 megatron 上根本不存在）。〔阶段一，随/在 W33 之后〕

**死参数（违反 fail-loudly）**
- **W20** GRPO `sapo tau_pos/tau_neg` 死参且 dev 默认 `tau_neg=1.05`≠twinkle `1.0`（`loss/grpo.py:430-437` 支持但未转发）。〔阶段三〕
- **W21** 离线 `rpo_alpha`/`ld_alpha`/`discopop_tau`/`reference_free`/`desirable_weight`/`undesirable_weight` 死参（twinkle `loss/dpo.py:132/133/137/170` 部分支持但 dev 从不转发）。〔阶段三〕
- **W24** 离线 `fit()` 收 `eval_dataloader`/`eval_steps` 却从不评估（eval 死线，`run_dpo.py:140-141/261-262`）。〔阶段三/四〕
- **W25** PPO `gamma`/`lam`/`cliprange_value`/`vf_coef`/`num_mini_batches`/`local_rollout_forward_batch_size`/`num_sample_generations`/`missing_eos_penalty` 全死参。〔阶段三〕
- 其余死参（`loss_weights`、`teacher_model_type`/`_revision`、`ref_model_type`/`_revision`、`mcore_ref_model`/`_adapter`、`f_divergence_type`、`real_tau`、`num_generations_eval`、`offload_bridge`、`router_replay_mode`、大量 `vllm_*`/`sglang_*` 引擎旋钮）：接线或 `_validate` fail-loudly 二选一。〔阶段三；`vllm_server_*` 按 §2.H 移除而非接线〕

**架构 / 复用 / 代码质量**
- **W28** GKD/GRPO 跨栈 import legacy `swift.rlhf_trainers.{gkd_helpers,vllm_client,utils}`、`swift.infer_engine.*`、`swift.rl_core.advantage.*`（`run_gkd.py:195/200/223/224/429/430`；`grpo.py:71-138/527`；`configure.py:467`）。按无-server 前提，HTTP 老师骨架**移除**；纯函数（`remap_teacher_logps_to_student_frame` 等）**下沉** twinkle。〔阶段三；阶段一新增方法不再新增此类 import〕
- **W29** GKD 老师抽象非多态（联合类型 + 散落 `isinstance`；多"老师"=按标签路由非加权）→ 直接挡住 MOPD 复用。**修**：按 §3.8(2) 提炼统一 `Teacher` 协议（`FrozenModelTeacher`/`DisableAdapterTeacher`/`MultiTeacher`），`score(features)->{logits|topk|多老师列表}`。〔阶段一，随 MOPD/OPSD 引入〕
- **W30** rollout `backend='vllm'` 硬编码（`run_grpo.py:261`），全量映射器 `build_engine_args`（`builders/sampler.py:126-143`）闲置，`_sampler_engine_args` 只转 4 旋钮；`plan_rl_device_groups` 未覆盖 sglang tp/pp/dp（B3）。〔阶段二〕
- **W14** 基类 `RolloutEngine` 静默跳过权重同步、无 warning（易误接未完成基类）。〔阶段二，随 `SyncableRollout` 归并修〕
- **W22** 离线全族无 backend fail-loudly；`padding_free` 对 dpo/kto 静默错。〔阶段三〕

**测试**
- **W31** 两份 `test_recipes.py` 并存（`feature/test_recipes.py` 163 行 e2e 接线表 vs `feature/rl/test_recipes.py` 995 行单元/张量）；`rl/test_recipes.py:711` 用已改名字段 `reward_funcs=`（应为 `orm`）必 TypeError；GRPO/PPO/GKD 均无经 `run_*` 入口的 full-loop e2e；opsd/mopd/rft dev 侧零覆盖。〔阶段四〕
- **W32** PPO 张量测试与实现漂移（GAE 用会被吞的 `gamma=/lam=`、`PPOValueLoss` 传 `outputs={'logits'}` 应为 `'values'`、`old_values=None` 触发硬断言）且本环境全 SKIP（`No module named twinkle`，importorskip 静默放行）→ 从未验证真实代码，掩盖 W03。〔阶段四；先修 twinkle 可导入性再让测试真跑〕

> **阶段 0 结论**：12 种算法中，**rm** 基本准确可用（仅缺 backend fail-loudly）；**cpo/orpo/simpo** loss 正确但共享 W01 静默训错；**dpo/kto** 另有 loss/权重缺陷；**grpo** 有 A1 崩溃 + gspo 降级 + completion_mask off-by-one + 大量死参；**ppo** 当前**完全跑不起来**（W03）；**gkd** 数学基座干净但老师抽象挡住 OPSD/MOPD 复用（W29）。三新方法（opsd/mopd/rft）dev 侧尚未实现。审计纯读未改任何代码。

### A. 真实 bug（会导致运行期崩溃或静默训错，优先修）
- **A1（=W00，已审计确认）**：`run_grpo.py:159` `CheckpointEngineManager(model=, sampler=, platform=, colocate=colocate)` —— twinkle 构造签名是 `mode: Literal['auto','naive','colocate','standalone']`（`checkpoint_engine/manager.py:69-75`），**没有 `colocate` 形参**，当前调用必然 TypeError；且 `mode` 从不传，即便删掉 `colocate=` 也会让同卡场景错落到 `standalone`（NCCL）。应改为 `mode=('colocate' if colocate else 'standalone')`。GRPO/PPO/RFT 全踩，无测试覆盖（`run_grpo` 零 e2e）。**阶段一必修。**
- **A2（=W02，已审计确认）**：`rollout/__init__.py:164` `completion_mask` 顺序 off-by-one（task #18）；`rollout/multi_turn.py:40-60` 已记录该 hazard，但多轮测试用全 1/对称 mask，未触发。RFT 复用 rollout 前必修。
- **A3（=W01，审计新发现，静默训错）**：离线偏好 `chosen_labels`/`rejected_labels` 从不做 next-token shift → dpo/kto/cpo/orpo/simpo 全部在自见分布上训练。**阶段一随三新方法的数据面一起修。**
- **A4（=W03，审计新发现，崩溃）**：PPO `run_ppo.py:433` 以位置参数调 kw-only 的 GAE → 首个 rollout 步必 TypeError，PPO 从未跑通；张量测试因 twinkle 不可导入全 SKIP 而掩盖了它。**阶段三 PPO 收敛批修。**

### B. 数据管线缺陷（算法正确性，非崩溃）
- **B1**：GRPO/PPO/GKD 每步重新生成整个 prompt 集，`num_train_epochs` 被忽略（只有 `max_steps` 生效）。
- **B2**：一次 `forward_backward` 只喂一个样本（one-sample-per-micro-batch），与 legacy 的 batch 语义/有效步长缩放存在差异。
- **B3**：`plan_rl_device_groups` 只覆盖 vllm 的 `tensor_parallel_size * data_parallel_size`；sglang 的 tp/pp/dp/ep 组合未纳入采样器 world size 计算。

### C. 声明但未接线的"死参数"（用户设了值却无效，违反 fail-loudly）
约 31 个，重点：`rpo_alpha`、`ld_alpha`、`discopop_tau`、`loss_weights`、`desirable_weight`/`undesirable_weight`、`async_generate`、`sleep_level`、`offload_optimizer`/`offload_model`、`generation_batch_size`/`steps_per_generation`、`prm*`（prm/prm_weights/prm_parallel_spec）、以及大量 `vllm_*`/`sglang_*` 引擎旋钮未转发进 RL rollout 采样器（`_sampler_engine_args` 只挑了 4 个）。
- **判据（reference.md 教训）**：死参数要么接线、要么在 `_validate`/warning 里显式点名"不生效 + 正确做法"，二选一，不允许静默。
- **"无 server"前提下的特殊处理（移除，不接线）**：`vllm_server_*`（base_url/host/port/timeout/group_port/pass_dataset）与 `teacher_model_server` 属于 HTTP server 语义，RL 训练路径**不起 HTTP server**，故这些参数应**移除**（连同 legacy `_RemoteGKDTeacher`/`_RemoteGRPOTeacher` 的 HTTP 抓取分支）；若为 CLI 平价暂留，则必须在 `_validate` fail-loudly 点名"训练路径无 HTTP server，sampler/teacher 走 Ray DeviceGroup"。详见 §2.H。

### H. "server"词汇清理（无 server / 全 Ray 架构前提的落地清单）
按文首架构前提，dev RL 训练路径下所有 HTTP server 语义要么改名、要么移除：
| 现状（HTTP/误名） | 位置 | v5 处理 | 阶段 |
|---|---|---|---|
| `vllm_mode='server'`（误名，实为分离式 Ray 设备组） | `rollout_config.py` / `run_grpo.plan_rl_device_groups` | 改名/重释为 `disaggregated`（与 `colocate` 相对）；保留 `'server'` 作 alias 但注释澄清非 HTTP | 二 |
| `vllm_server_*`（base_url/host/port/timeout/group_port/pass_dataset） | `rollout_config.py` | 移除；暂留则 `_validate` fail-loudly | 三 |
| `teacher_model_server` + `parse_teacher_model_server` + `TeacherServerConfig` | `rlhf_config.py` / legacy `gkd_helpers.py` | RL 训练路径移除；MOPD 多老师改为 Ray actor 冻结模型（§3.3） | 一（MOPD 不引 HTTP）/ 三（清 legacy） |
| `_RemoteGKDTeacher` / `_RemoteGRPOTeacher`（HTTP 老师） | `run_gkd.py` / `run_grpo.py` | 移除 HTTP 分支；老师 = 冻结 twinkle model actor，`forward_only` 打分 | 一/三 |
| `gkd_helpers` 的 HTTP gather/infer/scatter 路由 | legacy `swift/rlhf_trainers/gkd_helpers.py` | 不复用其 HTTP 骨架；仅借鉴 tag/加权语义 | — |
- **判据**：改名只动命令面/词汇，被复用的内部语义名保留；移除 HTTP 路径时确认无其他业务依赖（grep 全仓）。`swift deploy`/`swift infer` 的服务/离线 HTTP 路径不受影响。

### D. 跨栈 legacy import（dev RL 内部仍 import legacy swift）
`swift.rl_core.advantage.*`（SDAR/teacher-KL）、`swift.rlhf_trainers.{gkd_helpers, vllm_client, utils}`。这些是"通用底层能力"，按 `dev-module-authoring` 应下沉 twinkle 或迁进 dev 通用文件，而非跨业务/跨栈引用（阶段三处理，阶段一新增方法尽量不再新增此类 import）。

### E. 〔已重定性〕原「未实现能力」六项 → 接线/下沉工单（阶段五，见 §3.9）
经逐项回读 twinkle 与 legacy swift 源码核实：下列六项**没有一项是能力缺失**，全部是「twinkle 已原生支持（或 legacy swift 已实现待下沉）+ dev 侧接线缺口」——与辅助模型 backend 曾被误标 E 类是同一类错误。**移出「诚实记录的未实现能力」，落成 §3.9 的有范围工单**：
- **MoE 路由重放（R2/R3）**：transformers 侧 twinkle 已原生（`model/transformers/moe/router_replay.py` 的 `RECORD/REPLAY_FORWARD/REPLAY_BACKWARD`，`forward(router_replay_action=…)` + `routed_experts` 已在处理器 `align_routed_experts` 与放行名单）；megatron 侧（R3）twinkle grep=0，实现在 legacy `swift/megatron/utils/router_replay_utils.py` + `3rd/Megatron-LM/.../moe/router_replay.py`，按内核能力下沉 twinkle。**R2/R3 确切语义（已核实钉死，订正本 plan 早前的错误假设）**：`routed_experts` **可由采样器返回**——vLLM 引擎 `vllm_engine.py:497` `getattr(output,'routed_experts')` → `vllm_sampler.py:259-269` 注入 `new_input_feature['routed_experts']`（对应 vLLM PR 28284）。**R3** = 重放**生成期采样器捕获**的路由（「真正产出这些 token 的路由」，legacy `build_routed_experts_batch` 对 R3 断言 `routed_experts` 必在 rollout 数据里）；**R2** = 重放**训练模型 RECORD 前向**自采的路由（`is_r2_record_action`==RECORD），用于采样器不返回路由的场景。故 R3 无需额外 RECORD 前向（采样器已给），仅 R2 需要。
- **采样重放（sampling replay，与路由重放同族）**：twinkle 全链已通（vLLM `enable_return_sampling_mask`→`SampledSequence.sampling_mask`→`GRPOLoss(enable_sampling_replay=)`，约束 beta=0/entropy_coef=0），dev 仅缺透传接线。
- **RLHF 序列并行（Ulysses SP under ray）**：twinkle SP 计算原生（`strategy/sequence_parallel.py`），dev 缺 `build_ray_dp_mesh` 携带 ulysses 维 + validate 放开。
- **多模态**：twinkle 采样器与 transformers VL generate 全支持（vLLM `multi_modal_data`、sglang `_extract_image_data`、`generation.py` collate `pixel_values/image_grid_thw`、`VLM_CONCAT_FIELDS` 放行），dev `rollout/__init__.py` 却自限 "text-only"，缺 prompt 媒体透传 + 训练侧 vision 键流转。
- **异步生成（async_generate）**：twinkle 采样器是 async 引擎（`async def sample/batch_sample`），dev 缺 loop 双缓冲编排。
- **训练内 PRM**：`reward.py` 已有 prm 通道（infer 侧在用）+ `build_frozen_reward_model`/`compute_reward_model_scores`，dev RL 侧缺 PRM 作 frozen 辅助接入 per-token advantage。
- **padding_free + packing 全算法**：`forward_only` 已 `unpack_packed_sequences` 归一输出，validate 白名单只放行 grpo/dpo/kto/gkd，其余算法待逐一核实放开。
- **仍属真未实现（保留诚实记录，不在阶段五范围）**：外部 rollout server（无-server 前提，§2.H 已移除）、gym/env-scheduler（twinkle_agentic 有底层、dev 无消费者）。
- （**辅助模型 ref/teacher/reward/critic 的 megatron 后端不在此列**：按基本原则 1 它后端无关，只是构建点未透传 `backend` 的接线缺口，阶段三 s3aux 已修，见 §3.5。）

### F. 未用上的 twinkle 能力（复用机会）
- **原生数据类型/机制（D6 主线，阶段一起用）**：`Trajectory`/`Message`/`InputFeature`、`Trajectory.user_data`（reward/extras 透传）、`SampledSequence.new_input_feature`（rollout 免再 encode）、`concat_input_feature(appended_as=completion/demonstration/context)`（多轮/老师-学生共享 response/response-only 拼接）、`padding_free` InputProcessor。这些取代 dev 手搓的 shift/`completion_mask`/`replace_assistant_response_with_ids`。
- **其余复用机会**：`advantage/group_admission.py::{GroupAdmissionPolicy, SamplingBudgetController}`（cookbook `rl/grpo/group_admission.py` 为范本）、sampling replay（`sequence.sampling_mask` + `GRPOLoss(enable_sampling_replay=True)`，cookbook `grpo_sampling_replay.py`）、`twinkle.metric.*`（`CompletionRewardMetric`/`EmbeddingMetric`）、`twinkle.reward.*`、`LigerFusedLinearGRPOLoss`、MoE routing replay（`routed_experts` 字段）。
- **data plane（`asample_to_data_plane`/`DataRef`/`forward_backward_from_data_plane`）优先级下调**：cookbook 全部脚本均未使用（数据以 plain list-of-dicts 传递），故本轮不作为主线，仅记录为 twinkle 已具备但暂不采纳的能力。

### G. 既有测试破损（补测阶段先修）
`swift/dev/tests/feature/rl/test_recipes.py`、`feature/grpo/test_rollout.py` 存在签名漂移（`MultiTurnRollout` 签名、已删除的 `completion_length_limit_scope`、gym 路径移除、`reward_funcs`→`orm` 改名、gym reward 列不再追加）。

---

## 3. Design（分层放置 + 复用 + 下沉）

### 3.1 命令面 / 配置层
- **命令面不变**：仍是 `swift rl`（兼容 `swift rlhf`）。三种新方法通过 `rlhf_type` 暴露（见 D1）。
- **`swift/dev/config/rlhf_config.py`**：`rlhf_type` Literal 增加新值（D1 决定形态）；为三方法各加一组带 `#:` 注释的字段：
  - OPSD：复用既有 `teacher_model`/`teacher_adapters`/`_teacher_use_disable_adapter`/`beta`/`temperature`/`max_completion_length`/`lmbda`/`sft_alpha`；新增 `opsd_reverse: bool = True`（散度方向，对应 `OPSDLoss.reverse`）。数据侧特权上下文走 `teacher_prompt` 数据列（既有约定，无需新字段）。
  - MOPD：`teacher_model` 支持 `List[str]`（多个本地/Ray-actor 冻结老师，各占 DeviceGroup，无 HTTP）；新增 `teacher_weights: Optional[List[float]]`（加权聚合，D2-A）；`teacher_kl_coef`（既有）；`teacher_parallel_spec`/placement（各老师共享或各占设备组，§3.3）。**不引入 `teacher_model_server`**（无-server 前提，§2.H）。
  - RFT：`rft_num_samples: int = 8`（每 prompt 采样条数，可复用 `num_generations`）、`rft_select: Literal['threshold','best_of_n','top_k']`、`rft_threshold: float`、`rft_top_k: int`、`rft_iterations: int = 1`（迭代轮数）、`rft_max_samples_per_prompt: Optional[int]`。奖励复用既有 `orm`/`orm_weights`/`reward_model`。
- **`swift/dev/config/rollout_config.py`**：新增 `rollout_sampler: Literal['vllm','sglang'] = 'vllm'`（阶段二接线；阶段一先加字段 + `_validate` 对非 vllm fail-loudly 指向"阶段二"，避免死参数）。
- **`swift/dev/cli/rlhf.py`**：`field_owners` 补充新字段归属（`teacher_weights`/`teacher_parallel_spec`→RLHF；`rft_*`→RLHF；`rollout_sampler`→Rollout）；`RlhfCliCompatConfig`/`_apply_rlhf_compat` 若涉及新别名同步。
- **`swift/dev/cli/legacy_coverage.py`**：若新方法引入新的 legacy-only 分类键，按 backend 补 `CLI_LEGACY_ONLY`。

### 3.2 OPSD 实现（阶段一）
- **recipe**：新增 `swift/dev/recipe/run_opsd.py`（或在 `run_gkd.py` 内按 `distill_mode` 分支——见 D1）。核心循环 = GKDLoop 的"学生 `model.generate` → 老师 forward → 学生 `forward_backward`"骨架，但：
  - 老师输入用**特权视图**：从 `Trajectory.user_data` 的 `teacher_prompt` 构造 teacher messages，response token 与学生**共享**（同 id 同序）。**主线用 twinkle `concat_input_feature(..., appended_as=)`** 拼接"特权 prompt + 共享 response"，由 provenance 角色决定 labels/completion_mask（`template/base.py:230-268`），取代 dev 手搓的 `replace_assistant_response_with_ids`/`_teacher_feature` 断言（§1.5 A0-2）。
  - 老师 forward 取 **response-only 的 sampled-token logp**（`teacher_logps`），而非 GKD 的整词表 `teacher_logits`。
  - loss = twinkle `OPSDLoss`（已注册 `'opsd'`），通过 `configure_rlhf_loss` 选中；`forward_backward(teacher_logps=...)`。
- **老师来源（用户拍板：用「模型」不用「sampler」，因模型 logits 更稳定）**：`disable_lora`（LoRA 学生蒸馏自己的 base，即论文主设置）/ 独立冻结 `teacher_model`（Ray actor 冻结 twinkle model，`forward_only` 取 sampled-token logp）。**不含**远程 HTTP `teacher_model_server`（无-server 前提，§2.H 移除）；**不采用** cookbook GKD 的"第二个 sampler + `prompt_logprobs=topk`"路（`rl/gkd/gkd_on_policy.py:279-291`，已评估、因 logit 稳定性否决，见 D6）。
- **下沉 twinkle**：`OPSDLoss` 已在 twinkle，无需新增。老师-学生 response-token 对齐优先用 twinkle 原生 `concat_input_feature`/`SampledSequence.new_input_feature`；legacy `remap_teacher_logps_to_student_frame` 仅在原生原语不足以表达对齐时评估下沉。

### 3.3 MOPD 实现（阶段一）
- **recipe**：新增 `swift/dev/recipe/run_mopd.py`（或 OPSD/GKD 同族分支）。学生在线采样（同 OPSD，进程内 `model.generate`）；多老师对同一轨迹打分；聚合。
- **多老师来源（全 Ray actor 冻结「模型」，无 HTTP、非 sampler）**：`teacher_model: List[str]`，每个老师建一个**冻结 twinkle model actor**，各占独立 DeviceGroup：`build_model(teacher_cfg, DistributedConfig(mode='ray'), remote_group=f'teacher_{k}')`（镜像 `build_frozen_reward_model` 用 `remote_group` 把冻结 RM 放到自己 GPU 的做法）。老师打分走 `forward_only`（默认返回 label token 的 per-token logp，即 sampled-token 形式，正是 D5-A 所需），**不经 HTTP、不复用 legacy `gkd_helpers` 的 server 路由骨架**。用户拍板用「模型」而非 cookbook 的「第二个 sampler + prompt_logprobs」（D6：模型 logits 更稳定）。
- **设备组规划**：`plan_rl_device_groups` 需扩展为"trainer + (可选 sampler) + N 个 teacher group"的多组规划；MOPD 是进程内生成（无独立 sampler 权重同步），故组 = trainer + N teachers。各老师可共享一组（省卡，串行 `forward_only`）或各占一组（并发，费卡），由 `teacher_parallel_spec`/placement 配置决定（阶段一默认共享 trainer 之外的组，诚实记录并发为后续优化）。
- **聚合语义（D2-A = weighted）**：每个老师对每条样本产出 response-only `teacher_logps`（同 tokenizer 前提下 response token 逐位对齐，复用 `remap_teacher_logps_to_student_frame` 的对齐断言），按 `teacher_weights` 归一加权成聚合老师通道，喂下沉的 `MOPDLoss`。
- **融合数学（已实现，落定）**：K 个老师通道按**概率混合（mixture）**聚合成单一目标分布，在对数空间用 `logsumexp` 稳定计算：`teacher_logp_mix = logsumexp_k(teacher_logp_k + log w_k)`（即 `log(Σ_k w_k·p_k)`，`w_k≥0`、`Σw_k=1`；`w_k=0` 的老师跳过以免 `log0=-inf`，归一前的正和校验保证至少一路存活）。随后复用 OPSD 的 sampled-token k3 估计（`r = teacher_logp_mix − student_logp`，`exp(r)−r−1`）+ BNPO token-mean 聚合。**为什么选混合而非加权和/几何平均**：混合给出一个良定义的单一目标分布、loss 尺度不随 K 增长（符合"聚合成一个 token 级蒸馏目标"）；几何平均（`Σ_k w_k·teacher_logp_k`）会把目标推向 product-of-experts，更尖锐、易被最自信的老师主导，老师们分歧时混合更宽容。
- **下沉 twinkle（已完成）**：新增 `twinkle/src/twinkle/loss/mopd.py::MOPDLoss(OPSDLoss)`（接收 `List[teacher_logps]` + `teacher_weights`，内部归一加权后复用 OPSD 的 k3 估计；单老师/裸张量输入委托 `super().__call__`，是 OPSD 的严格超集），注册 `torch_loss_mapping['mopd']`。为让 MOPD 复用同一份学生 logps 与 k3 核心，`opsd.py` 已抽出两个行为保持的 helper：`_student_logps_and_mask(inputs, outputs)` 与 `_per_token_distill_loss(teacher, logps)`（OPSD 自身 `__call__` 逐字节等价）。配套 `twinkle/tests/loss/test_mopd.py` 留补测阶段。
- **不复用/移除**：legacy `TeacherServerConfig`/`parse_teacher_model_server`/`route_samples_to_teachers`/`fetch_teacher_parsed_by_routing`（HTTP）在 MOPD 不使用；`route` 语义（按 tag 每样本一老师）在 D2-A 下不实现（若将来需要，作为 `MOPDLoss` 的可选聚合模式另议，非本轮）。

### 3.4 拒绝采样微调 RFT 实现（阶段一）
- **recipe**：新增 `swift/dev/recipe/run_rft.py` + `RFTLoop`（peer of `GRPOLoop`/`GKDLoop`）。每轮：
  1. **rollout**：复用 `SamplerRollout`（含权重同步 + colocate 内存调度）对每个 prompt 采 `rft_num_samples` 条（`SamplingParams.num_samples` 或重复 prompt）；**直接取 `SampledSequence.new_input_feature`** 作可训练特征（`concat_input_feature` 已填 labels+completion_mask，免手搓 shift，§1.5 A0-2）。
  2. **打分**：复用 `reward.py`——`get_reward_funcs(orm)` + `compute_rewards_per_func` + `weight_rewards`；奖励模型走 `build_frozen_reward_model`/`build_reward_model_plugins`/`compute_reward_model_scores`；打分所需的 `solution`/`ground_truth` 等从 `Trajectory.user_data` 透传。
  3. **过滤**：按 `rft_select` 选子集（threshold：`reward >= rft_threshold`；best_of_n：每 prompt 取最高；top_k：每 prompt 取前 `rft_top_k`）。空子集 fail-loudly。
  4. **SFT**：选中样本的 `new_input_feature` 已是 response-only、labels/completion_mask 就绪（twinkle 原生约定，等价 SFT 的 next-token 监督），直接 `configure_loss(cross_entropy)` + `forward_backward`（复用 SFT 前向反向）。
  5. **迭代**：`rft_iterations` 轮，轮间策略权重同步进采样器（复用 `SamplerRollout.sync_weights`）。
- **无新 loss、无新 twinkle 原语**：RFT 是既有原语（`SampledSequence.new_input_feature` + reward + cross_entropy）的组合，是三者中最干净的。
- **放置**：`run_rft.py` 只放 RFT 特有的"采样→打分→过滤→SFT"编排；打分/过滤的通用小工具（如 best-of-N 选择）若 2+ 业务会用则放 `swift/dev/` 通用文件，否则留在 recipe 私有 helper。

### 3.5 通用化：backend × sampler × 放置（阶段二为主）
- **策略模型后端**（transformers/megatron）：`DistributedConfig.backend` + `TrainAssembly.build_model` 统一的是**训练/前向**接口——`forward_only`/`forward_backward`/`calculate_loss` 是 `TrainableModel` 基类的 `@abstractmethod`、两后端都实现，故打分/训练/老师前向天然后端无关。**但 `generate` 不在统一接口里**（基类无此抽象方法）：它是后端特化的生成能力，`TransformersModel` 有就地实现、`MegatronModel` 按设计拒绝（见下条与 W34）。所以「三新方法天然继承后端通用性」只对训练/前向成立、**对生成不成立**——这是原 plan 把蒸馏生成当作后端无关的根因。unsloth = `TunerConfig.tuner_backend='unsloth'`（tuner 层，已统一）；liger = `LigerFusedLinearGRPOLoss`/`liger_fused_linear_cross_entropy`（loss 层，已注册），RFT 的 SFT 段可选 liger fused CE。
- **rollout 采样器**（vllm/sglang/进程内）：
  - GRPO/PPO/RFT 用**独立进程采样器 + 权重同步** → 只能 vllm/sglang（二者都有 `CheckpointEngineMixin`）。接缝：把 `run_grpo.py:262` 的 `backend='vllm'` 改为读 `rollout_config.rollout_sampler`，走既有 `build_sampler`；`_sampler_engine_args` 按 backend 选 `vllm_*`/`sglang_*` 前缀（复用 `build_engine_args` 的前缀剥离）；`plan_rl_device_groups` 的 `sampler_world_size` 按 backend 计算（vllm: tp*dp；sglang: tp*pp*dp，修 B3）。
  - **transformers 不能做独立进程权重同步**（`TransformersSampler` 无 `CheckpointEngineMixin`，docstring 明确"故意不提供"）。因此 `rollout_sampler='transformers'` 对 GRPO/PPO/RFT 必须 fail-loudly，指向 vllm/sglang。
  - **进程内 generate 是 transformers 专属、非后端通用（修正原「transformers 或 megatron 皆可」的错误断言 → W34）**：GKD/OPSD/MOPD 现状用 `TransformersModel.generate`（`model/transformers/transformers.py:888`，`@remote_function(dispatch='slice_dp', collect=collect_dp_and_flatten)`，把 prompts 切给各 data rank、每 rank 用自己常驻权重就地生成分片）在训练模型自己的 worker 上就地生成，无第二份权重、无权重同步，故 transformers 下天然 DP>1（W33）。**但 `MegatronModel.generate`/`generate_stream` 按设计 `raise NotImplementedError`**（`model/megatron/megatron.py:450-472`：megatron 权重按 TP/PP 分片、用 megatron 自己的参数名，无推理引擎能读；错误信息直接指向「用独立采样器 vLLM/SGLang/Transformers 加载自己那份权重」）。而 `DistributedConfig.backend` 接受 `'megatron'`、蒸馏策略经同一 `assembly.build_model()`（`run_gkd.py:107`）构建 → **megatron 策略的蒸馏配置可达、并在首个 `self.model.generate`（`run_gkd.py:402`）处崩，当前无前置守卫**（`cli/rlhf.py` 仅 `--seq_kd` 一处 raise）。修法见 W34：①`megatron × 进程内 generate` 须在校验期前置 fail-loudly 指向 vllm/sglang；②〔用户拍板〕蒸馏学生 rollout **一律**走与 GRPO/RFT 同一套 `Rollout` 协议（独立采样器 vllm/sglang + 逐步权重同步；`TransformersModel`/`MegatronModel` 都混入 `CheckpointEngineMixin`，故采样器路两后端皆通），**不再用就地 generate 作 rollout**；就地 generate 只保留给训练中间 eval（transformers-only；有采样器时 eval 复用采样器）。老师打分是 `forward_only`（统一接口、两后端皆通），继续走冻结 model actor。
  - **关键澄清（修 W33 的认知误区 + 用户拍板）**：蒸馏曾误用「就地生成 + `mode='local'` DP=1」，两处都错——放置须走 `mode='ray'` DP mesh（§1.4 硬契约，W33），生成须走独立采样器 + 权重同步（与 GRPO/RFT 同一 `Rollout` 协议，后端通用，W34）。**就地生成仅 transformers 可行（`MegatronModel.generate` raise），且按用户决策不再用于 rollout，只保留给训练中间 eval（transformers-only）。蒸馏由此与 GRPO 结构同构：Ray DP + 采样器 rollout + 冻结老师 actor（`forward_only`）+ 蒸馏 loss。**
- **放置**（colocate/disaggregated）：`plan_rl_device_groups` + `CheckpointEngineManager(mode=)` 已覆盖 vllm；阶段二把 sglang 的 world size 与 sleep/wake_up 语义纳入（sglang 引擎已有 `update_weights`/`sleep`/`wake_up`）。**术语澄清**：`vllm_mode` 的 `'server'` 是历史误名，实际是"分离式(disaggregated) Ray 设备组"（trainer/sampler 占不相交 GPU、NCCL 权重同步），与 `'colocate'`（共享 GPU、CUDA IPC 权重同步）相对；阶段二把它改名/重释为 `disaggregated`（`'server'` 保留为 alias），`CheckpointEngineManager` 对应 `mode='standalone'`（disaggregated）/`'colocate'`。MOPD 的多老师 DeviceGroup 规划也在此扩展（§3.3）。
- **辅助模型后端与放置**（ref/teacher/reward/critic）：按**基本原则 1**，辅助模型只做 `forward_only`（`TrainableModel` 统一接口、两后端皆实现），故**后端无关、megatron 与 transformers 行为完全一致**，不是 E 类未实现能力。**〔状态：阶段三 s3aux/s3dpo 已修，见 RL_PROGRESS 同名条目〕** 原缺陷：在线 RL（grpo/ppo/rft）的 ref/reward/teacher 与离线全量微调 dpo/kto 的 ref，构建点（`run_ppo._build_reference`/`_build_reward_models`、`run_grpo._build_frozen_model`/`_build_reward_model_scorers`、`run_dpo._load_frozen_reference`、`_distill.build_frozen_teacher`）曾用**不带 `backend` 的 `DistributedConfig(mode='local')`** 建成 driver 进程内模型——而 `mode='ray'` 下 driver 不持 GPU，故这些路径其实从未被冒烟触发。现由共享原语 `builders.frozen_auxiliary_distributed_config(distributed_config, world_size, *, parallel_spec, deepspeed)` 统一修掉（继承 backend+bridge_backend、强制 mode='ray'、按 world_size 定 nproc）：
  - **① 后端透传**：辅助模型的 `DistributedConfig` 继承 run 的 `backend`+`bridge_backend`；辅助模型是各自独立进程、各自 `mpu` 的 Ray actor，与 megatron 策略不争进程级 `parallel_state` 单例，故 megatron 策略 + megatron 辅助模型天然共存。
  - **② 放置（用户拍板 2026-10）**：按**基本原则 2** 辅助模型恒 `mode='ray'`、构建点残留的 `mode='local'` 一律去掉；占卡规则按辅助模型形态分派——**LoRA 形态**（`disable_lora` 参考、挂在策略 base 上的适配器）复用策略已加载权重、不是独立副本，**可与策略模型或 sampler 共卡、不需新设备组**（现有 `disable_lora` 路径天然正确）；**全参形态**（全量微调独立 `ref_model`、seq_cls 奖励模型、独立 RLSD/SDAR teacher）是独立权重副本，**必须异构、各占一个不相交 DeviceGroup**（镜像蒸馏老师组与 `run_infer` 的 orm/prm 组范式：`plan_rl_device_groups` 扩 `auxiliary_groups` 为每个全参辅助模型追加一个独占组，默认 1 rank、teacher 可选 `teacher_parallel_spec`；`assembly.initialize_twinkle` 同扩 `auxiliary_groups` 供离线族；构建点用 `build_model(cfg, frozen_auxiliary_distributed_config(...), remote_group=<该组>)`）。判定与构建同源（`_frozen_model_needs_group`/`_reference_needs_group` 镜像构建分支，防漂移）。**诚实记录的资源画像**：全参辅助模型在 ray 下各占自己的 GPU（不再像 local 与策略挤同卡），总卡数 = trainer + (sampler if disaggregated) + 每个全参辅助模型组。
  - MOPD/OPSD 老师已按"无 server"前提改为 Ray actor 冻结模型（`mode='ray'` + `remote_group`，§3.3），不再是 legacy 的 HTTP 远程老师；蒸馏老师 `_distill.build_frozen_teacher` 现经 `frozen_auxiliary_distributed_config` 透传 run backend（s3aux 已修，不再是不带 backend 的形态）。

### 3.6 需要下沉 twinkle 的清单（通用底层能力）
| 能力 | 现状 | 下沉目标 | 阶段 |
|---|---|---|---|
| OPSD loss | 已在 `twinkle/loss/opsd.py` | 无需动 | — |
| MOPD loss（多老师加权） | 无 | `twinkle/loss/mopd.py::MOPDLoss(OPSDLoss)` + 注册 + 测试 | 一（D2-A weighted） |
| 多老师 Ray actor 冻结模型 + `forward_only` 打分 | legacy 是 HTTP 路由（`gkd_helpers`），**不复用** | dev 侧按 `build_frozen_reward_model` 的 `remote_group` 模式建 Ray actor；无 HTTP 骨架需下沉 | 一 |
| teacher-logp 帧对齐 | legacy `remap_teacher_logps_to_student_frame`（纯函数） | 评估下沉 twinkle 工具（跨栈 import 收口） | 三 |
| rollout 采样器 backend 选择 | dev `build_sampler`/`_derive_sampler_type`（infer/deploy 已用） | 复用到 RL rollout（不下沉，dev 内复用） | 二 |
| HTTP server 语义清理 | `vllm_mode='server'`/`vllm_server_*`/`teacher_model_server`/`_Remote*Teacher` | 改名 disaggregated / 移除（§2.H） | 二/三 |

### 3.7 复用资产清单（不重写）
- **twinkle 原生数据类型/机制（主线，§1.5 A0）**：`Trajectory`/`Message`/`InputFeature`（`data_format/`）、`Trajectory.user_data`（reward/extras 透传）、`SampledSequence.new_input_feature`（rollout 免再 encode）、`concat_input_feature(appended_as=)`（老师-学生共享 response / 多轮 / response-only 拼接）、`set_template`/`set_processor(padding_free=)`/`encode(add_generation_prompt=True)`、`CheckpointEngineManager.sync_weights`。
- **swift/dev 既有**：rollout `SamplerRollout`、`plan_rl_device_groups`、`_sampler_engine_args`、`_grpo_sampling_params`、`_prompt_rows_from_dataset`（改为产出 `Trajectory`）。
- 打分：`reward.py` 全套 + `rewards/orm.py`（8 规则奖励）。
- 老师/冻结模型：`_build_teacher`、`_build_frozen_model`、`_build_reward_model_scorers`、`configure_frozen_adapter`（`_RemoteGKDTeacher`/`_RemoteGRPOTeacher` 是 HTTP 老师，按无-server 前提**移除而非复用**，见 §2.H；dev 私有的 `SHIFTED_KEY`/手搓 shift 由 twinkle 原生 `concat_input_feature` 取代）。
- 装配：`TrainAssembly`、`configure_rlhf_loss`/`configure_loss`、`configure_optimizer`/`resolve_max_grad_norm`、`RunTracker`、`save_training_checkpoint`。
- loss：twinkle `torch_loss_mapping`（opsd/grpo/bnpo/cross_entropy/infonce/pointwise_reranker/listwise_reranker/liger_*）。

### 3.8 类与复用设计（好理解 / 好定制 / 好调试）

**现状问题（阶段 0 审计已见）**：`SFTLoop`（`train_loop.py:120`）是功能齐全的富 loop（step 计数、tracker、checkpoint、manual-gc、GA 边界、resume、callbacks、eval/best-model/hub）；但 `GRPOLoop`（`grpo.py:166`）/`PPOLoop`（`run_ppo.py:281`）/`GKDLoop`（`run_gkd.py:264`）/`PreferenceLoop`（`run_dpo.py:208`）**各自独立**，把上述脚手架重复实现（`GRPOLoop.__init__:228-248` 与 `SFTLoop` 的 checkpoint/logging/gc 块几乎逐行相同），`fit()`/`resume()` 各写一份。新增 3 算法若继续复制，会有 8 份漂移的脚手架——难懂、难定制、难调试。

**设计目标 → 三条手段**：
- **好理解** = 一条 loop 主干（cadence）只写一次；每个算法 = 主干 + 少量命名清晰的 hook。
- **好定制** = 定制点集中在 hook 与被注入的协作对象（Rollout/Teacher/Reward/Advantage），改一个算法不碰别的；协作对象是组合（注入），不是继承进 loop。
- **好调试** = 每步的并行数据装进**一个值对象**（不再有"几条平行 list 靠索引对齐"——task #18 的 off-by-one 正源于此）；协作对象各自可单独驱动/打桩。

**（1）Loop 类层次（模板方法：基类持有 cadence，子类覆盖 hook）**
```
TrainLoop                     # 主干：global_step/micro_step、GA 边界、tracker、checkpoint(save/resume)、
                              #        manual-gc、callbacks、history。fit()/resume() 只此一份。
├─ SFTLoop                    # 监督：dataloader 迭代 → forward_backward（已存在，改为继承 TrainLoop）
├─ PreferenceLoop             # 离线偏好：pair → 偏好 loss（dpo/kto/cpo/orpo/simpo/rm，无 rollout）
└─ OnPolicyLoop               # 在线主干：持有 Rollout + Reward + 权重同步节奏；
                              #   fit 每步：rollout → score → training_step(batch) → log/save
   ├─ GRPOLoop                #   hook: compute_advantages + policy-gradient loss
   ├─ PPOLoop                 #   加 critic/value（GAE、value forward_backward）
   ├─ RFTLoop（新）           #   hook: select_subset(reward 过滤) → 复用 SFT 的 cross_entropy 段
   └─ DistillLoop             #   蒸馏主干：学生 rollout + Teacher 打分；hook: teacher_logps
      ├─ GKDLoop              #     整词表 top-k logits + JSD（现状保留）
      ├─ OPSDLoop（新）       #     Teacher=自身/base 模型，sampled-token logp + OPSDLoss
      └─ MOPDLoop（新）       #     MultiTeacher 加权 + MOPDLoss
```
- **基类只放"所有 loop 都一样"的部分**（step/save/log/gc/resume/callback）；`OnPolicyLoop` 只放"所有在线算法都一样"的部分（rollout 调用、权重同步 timing、reward 打分、把结果装进 batch 值对象）；算法差异全落到命名 hook（`compute_advantages`/`select_subset`/`teacher_logps`/`training_features`）。
- **定制一个新 RL 算法 = 继承最近的基类 + 覆盖 1~2 个 hook**，不复制 fit()。这是"好定制"的核心。

**（2）被注入的协作对象（组合，各自可独立调试/替换）**
| 协作对象 | 职责 | 复用/来源 | 定制点 |
|---|---|---|---|
| `Rollout`（协议）| prompt → 采样 → 可训练特征 | 基类 `RolloutEngine`（twinkle `sampler`+`Template`，token-in-token-out，直接取 `SampledSequence.new_input_feature`）；`SyncableRollout` 子类加 `CheckpointEngineManager` 权重同步 + colocate 内存调度（现 `SamplerRollout` 归并入此） | backend vllm/sglang（阶段二）；多轮 `MultiTurnRollout` |
| `Teacher`（协议）| 给 response-only per-token logp | `FrozenModelTeacher`（冻结 twinkle model actor + `forward_only`，D6）/ `DisableAdapterTeacher`（策略 base，`disable_lora`）/ `MultiTeacher`（加权 List，MOPD） | 换老师来源不改 loop；**无 HTTP 实现** |
| `Reward`（协议）| completions + user_data → 分数 | `reward.py` 全套（`get_reward_funcs`/`compute_rewards_per_func`/`weight_rewards` + reward-model plugins） | 规则奖励/RM 插件可插拔 |
| `Advantage`（协议）| rewards → advantages/returns | `advantage.py`（grpo/gspo/…）+ PPO 的 GAE | estimator 可换 |

**（3）每步值对象 `RolloutBatch`（消灭平行 list，好调试）**
- 一个 dataclass 持有本步所有对齐字段：`features: List[InputFeature]`、`completion_mask`、`old_logps`、`rewards`、`advantages`/`returns`、`teacher_logps`（蒸馏）、`prompt_id`/`user_data`（透传）。所有"按样本对齐"的操作只在这一个对象上做，长度不一致在构造时即 fail-loudly（而非在 loss 里静默错位）。
- 取代现状：`GRPOLoop` 里 samples/old_logps/advantages 等分散在多个平行 list + `sample.prompt_id` 字符串索引（`grpo.py:271-292`）。

**（4）RL 仅 Ray，不支持 torchrun（用户明确）**
- RL 训练路径**只在 `DistributedConfig.mode='ray'` 下运行**（`run_grpo._initialize_twinkle_rl` 已强制）；driver 是唯一 loop 拥有者，模型/采样器/老师都是 Ray actor。
- 因此 `OnPolicyLoop` 及其子类**不带 torchrun 的 rank 守卫分支**（`SFTLoop._is_main_process` 那类 `dist` fork 只留在 SFT/`TrainLoop` 的 torchrun 兼容路径，RL 子类走 Ray 单驱动，天然无需）。这简化 RL loop、去掉一整类"多 rank 各跑一份 loop"的调试陷阱。
- 校验：RL recipe 在 `mode!='ray'` 时 fail-loudly 指向"RL 用 Ray"。**〔W33 注〕此守卫当前并不覆盖蒸馏**——gkd/opsd/mopd 被 `cli/rlhf.py` 留在 `mode='local'` 且不报错，正是本契约被违反之处；W33 修法要么把蒸馏翻转为 `mode='ray'`（首选），要么在该守卫下 fail-loudly，二者必居其一，不允许蒸馏静默跑 DP=1。**〔W34 注〕另一条正交守卫**：`backend='megatron' × 进程内 generate` 是不可行组合（`MegatronModel.generate` 按设计 raise），须在 `validate.py`（`_check_rollout_sampler`/`_check_distillation`）前置 fail-loudly 指向 vllm/sglang，不许拖到运行期首个 rollout 步崩；蒸馏的后端通用生成走 `Rollout` 协议（W34），与 W22（离线偏好无 backend fail-loudly）同属「不可行组合须前置报错」一类。

**（5）落地方式（剃刀，不推倒重来）**
- 阶段一先抽 `TrainLoop` 主干 + `OnPolicyLoop`/`DistillLoop` 两级基类，把 `SFTLoop`/`GRPOLoop`/`GKDLoop` 收敛为子类（行为不变、只搬脚手架），再加 `RFTLoop`/`OPSDLoop`/`MOPDLoop`。`PreferenceLoop`/`PPOLoop` 的收敛放阶段三/四（避免阶段一改动面过大）。
- `RolloutBatch` 值对象与 `Rollout`/`Teacher` 协议阶段一随三新方法引入；`SamplerRollout`→`SyncableRollout` 归并在阶段二（采样器通用化）一起做。
- 每次抽取以"行为等价"为不变量：抽完跑既有 e2e 须与抽前一致（review-mode pass 核对，§5）。

**（6）step-1b 具体落地子步骤（行为等价；每子步独立 e2e 复核后再进下一步）**

现状：`SFTLoop`(train_loop.py)/`GRPOLoop`(grpo.py)/`GKDLoop`(run_gkd.py) 各带一份近乎重复的脚手架（`__init__` 尾部计数器/tracker/save 旋钮、`_record_step`、`save`、`resume`、`_is_grad_sync_boundary`、fit 的 gc 括号），OPSD/MOPD/RFT 已继承 GKD/GRPO。三者的 `_record_step`/`save`/`resume` 有**真实 drift**（下），故抽取必须走「模板方法 + hook」而非直接上提，否则会给 GRPO/GKD 平白加上 SFT 才有的 callbacks/eval/hub（=改行为）。

**drift 清单（决定 hook 边界，全部来自逐行读三个 loop）**：
- `_record_step` 额外指标：SFT=`grad_norm` + `loss_*` channel + mtp；GRPO=`entropy` + `rollout_log_ratio`；GKD=`grad_norm`。→ hook `_extra_step_metrics(metrics)->dict`（默认 `{}`）。
- `_record_step` 步后动作：SFT=`_sync_state` + eval(`_due_for_eval`) + callbacks(`on_step_end`/`control.should_save`/`should_evaluate`)；GRPO=`_sync_reference`；GKD=无。→ hook `_post_step()`（默认 no-op）。
- `should_log`：SFT/GKD=`tracker.should_log if logging_config else (logging_steps 取模)`；GRPO=恒 `tracker.should_log`。→ hook `_should_log()`（默认 SFT/GKD 形，GRPO 覆盖以保持逐位等价）。
- `save` 的 consumed 源：SFT=dataloader `get_state()['consumed_train_samples']` + `on_save` callback + hub push；GRPO/GKD=`global_step`、无 callback/hub。→ hook `_consumed_train_samples()`（默认 `global_step`）+ SFT 覆盖 `save` 挂 callback/hub。
- `resume`：SFT=对齐 micro/global + dataloader skip + `_start_epoch`；GRPO/GKD=仅 `micro_step=cur_step`、`global_step=consumed`。→ 基类 `resume` 取 GRPO/GKD 形，SFT 覆盖加 dataloader skip。

**子步骤（每个都「只搬不改」+ e2e 复核）**：
- **1b-A `TrainLoop` 主干**：新基类置于 `train_loop.py` 顶部（模块 helper `is_grad_sync_boundary`/`num_optimizer_steps`/`start|collect|finish_manual_gc`/`save_training_checkpoint` 原地不动，避免 import churn）。持有公共 `__init__`（model/ga/max_grad_norm/max_steps/logging/output_dir/save 旋钮/manual_gc(+`<0` guard)/global_step/micro_step/history/tracker）+ `_is_grad_sync_boundary`/`_reached_max` + 模板 `_record_step`（公共 cadence + 上述 4 个 hook）+ 公共 `save`（走 `_consumed_train_samples` hook）+ 公共 `resume` + fit 的 gc 括号 helper。把 `SFTLoop`/`GRPOLoop`/`GKDLoop` 改为 `TrainLoop` 子类，各自只留 drift hook 覆盖（SFT 另保留 eval/best-model/megatron GA/dataloader/callbacks/hub）。**等价判据**：三 loop 的 `_record_step`/`save`/`resume` 逐行 diff 只体现为「公共段进基类、drift 段进 hook」，无语义改动。
- **1b-B `RolloutBatch` 值对象**（GRPO/RFT 面）：一个 dataclass 持有本步全部对齐字段（`features`/`completion_mask`/`old_logps`/`rollout_logps`/`ref_logps`/`teacher_logps`/`advantages`/`truncated`/`prompt_id`/`user_data`），构造时校验各 list 等长（fail-loudly，消灭 task#18 那类平行 list off-by-one）。替换 `_rollout_step` 返回的 per-sample dict 与 `_mini_batch_kwargs` 的平行 list 组装；`RFTLoop._select_samples`/`fit` 同步改为在 `RolloutBatch` 上切片。
- **1b-C `Teacher` 协议（解 W29）**：把 `{'disable_lora' 哨兵, 冻结 model actor, _RemoteGKDTeacher/_RemoteGRPOTeacher}` 联合类型 + 散落 `isinstance`，收敛为 `Teacher` 协议 + `DisableAdapterTeacher`/`FrozenModelTeacher`/`DynamicSelfTeacher`/`MultiTeacher`（加权 K 老师），统一 `score(features, view)->{logits|logps|channels}`。GKD/OPSD/MOPD 的 `_teacher_kwargs`/`_teacher_logits`/`_teacher_response_logps` 与 grpo.py 的 `_teacher_feature`/`_teacher_logps`/`_response_logps` 全改走协议；`_distill.py` 成唯一老师视图原语（grpo.py 重复 helper 并入，清掉 `_distill.py:16-19` 的 DUPLICATION 注记）。HTTP `_Remote*Teacher` 暂留为协议的一个实现（阶段三删）。
- **1b-D `OnPolicyLoop`/`DistillLoop` 中间基类**：`OnPolicyLoop(TrainLoop)` 持 Rollout+Reward+权重同步 cadence（fit：rollout→score→training_step(mini_batches)→log/save），`GRPOLoop`/`RFTLoop` 收敛为子类（GRPO hook=`compute_advantages`+policy-gradient loss；RFT hook=`select_subset`+cross_entropy）。`DistillLoop(OnPolicyLoop)` 持 student-generate→teacher-score→student-forward_backward cadence，`GKD`/`OPSD`/`MOPD` 收敛为子类（hook=teacher score + `_forward_kwargs`）。`PPOLoop`/`PreferenceLoop` 留阶段三/四。
- **1b-E colocate 交接 dedup**：把 `SamplerRollout.sync_weights/finish_generate` 与 `assembly.eval_enter/eval_exit` 重复的「wake weights→sync→offload trainer→wake kv_cache」/「sleep sampler→reload trainer」序列抽成单一 `ColocateHandover` 原语（两处共用，Bug#8 只需改一处）。`SamplerRollout→SyncableRollout` 的后端通用化仍留阶段二（§3.8(5)）。

**行为等价复核（每子步后跑）**：既有 e2e 冒烟（SFT + GRPO/GKD/OPSD/MOPD/RFT DP=1 + RFT/GRPO DP=2），固定 seed 下 **loss 序列 + checkpoint 须与抽前逐位一致**（确定性 loss 曲线即 oracle）。review-mode pass 核对「只搬不改」。**先做 1b-A**，复核通过再进 1b-B。**注（W33）**：1b 等价门禁下蒸馏家族仍按现状 DP=1 复核（DP=1 是 DP>1 的退化基线）；蒸馏的 **DP=2 复核随 W33 在 1b 之后补**（届时蒸馏走 `mode='ray'`、`batch_size=per_device×dp_size`、老师 Ray actor，须 DP=1+DP=2 双复核）。

### 3.9 阶段五：原 E 类六项 → 接线/下沉工单（用户明确要求做；一次设计、按依赖序实现）

> 前提：六项经源码核实无一是能力缺失（见 §2.E）。两条绝对原则不变（RL 恒 ray、后端等价仅 generate 有别）；megatron 路由重放下沉 twinkle 正是原则 1（两后端能力对齐）的落实。跨切面约束：megatron 路由重放属内核能力 → 下沉 twinkle、不在 dev 侧子类/patch 规避，补丁只用模型自带 `apply_patch`；新增 frozen 辅助（PRM）按 LoRA 共卡/全参异构独占组分档；所有不可行组合前置 fail-loudly，不静默降级。

**实施顺序（有依赖序）**：① twinkle 底层（W35 megatron 路由重放下沉 + W36 `build_ray_dp_mesh` 携带 ulysses）→ ② dev 接线（W37 R2/R3+采样重放、W38 SP 放开、W39 多模态、W41 PRM、W42 padding_free 放开）→ ③ W40 异步生成最后（依赖 replay 机制与 staleness 决策）。全程 AST 校验；review-mode pass 对真实代码执行。

- **W35 MoE 路由重放（R2/R3）+ 采样重放**：
  - **W35.1 R3（megatron）下沉 twinkle**：新建 `twinkle/src/twinkle/model/megatron/moe/router_replay.py`，镜像 transformers 侧接口，把 legacy `swift/megatron/utils/router_replay_utils.py`（`get_local_topk_idx_for_current_rank` 的 pp/cp/sp 切片、`RouterReplayHelper`、`apply_router_replay_patch` 的 `MoEAlltoAllTokenDispatcher.preprocess` patch）迁入，依赖 mcore `megatron.core.transformer.moe.router_replay.RouterReplay`（≥0.16）；由 `MegatronModel.forward` 按 `enable_router_replay`/`router_replay_action` 驱动（对称 transformers.py:687-694）；`apply_router_replay_patch` 走模型自带 apply_patch 一等入口。
  - **W35.2 R2/R3 dev 接线（transformers 优先）**：模型构建按 `router_replay_mode != 'disabled'` 开 `enable_router_replay`；**R3** 直接消费采样器回传的 `routed_experts`（已在 `SampledSequence`/`new_input_feature`/处理器 `align_routed_experts`），存入 `RolloutBatch`，训练 `forward_backward(router_replay_action=REPLAY_FORWARD, …)` + backward 自动 REPLAY_BACKWARD（twinkle 已内建，transformers.py:1113-1117）；**R2** 在训练前对 rollout token 跑一次 `forward_only(router_replay_action=RECORD)` 捕获 `routed_experts` 再走同一 REPLAY 路。`router_replay_mode` 从死参（现挂 `_UNWIRED_RLHF_KNOBS`）改为接线，不可行组合 fail-loudly（如 megatron×R3 未下沉前、采样器不回 routed_experts×R3）。
  - **W35.3 采样重放接线**：`_sampler_engine_args` 透传 `enable_return_sampling_mask`；`RolloutSample`/`RolloutBatch` 携带 `sampling_mask`；GRPO 构建 loss 时按开关传 `enable_sampling_replay=True`（约束 beta=0/entropy_coef=0，违反 fail-loudly）。
  - **开放决策（须用户拍板）**：R2 的 RECORD 前向额外一次全序列前向的显存/时序成本是否可接受；采样重放是否本轮纳入（推荐纳入，成本极低）。
- **W36 RLHF 序列并行（Ulysses SP under ray）**：`build_ray_dp_mesh` 增 `ulysses_size` 参 → `DeviceMesh.from_sizes(world_size=nproc, dp_size=nproc, ulysses_size=sp)`；`_apply_ray_placement` 用它；colocate 采样器共享同一 mesh。`train_batch_size = per_device × data_world_size`（读 mesh 的 `data_world_size` 自动 = world/ulysses，GRPO/PPO 已如此读、公式不改）。放开 validate：`_check_rlhf_sequence_parallel` 全 RLHF 硬 raise 改为按可行性守卫（复用 `_check_hf_sequence_parallel` 的 flash-attn / padding_side=right / world%sp==0）；megatron TP-SP（`DistributedConfig.sequence_parallel`，已在 `build_device_mesh` 接线）是另一维、不动。**开放决策**：SP×packing×padding_free 组合矩阵的可行边界。
- **W39 多模态（整体支持）**：去掉 dev `rollout/__init__.py` 的 "text-only" 自限；`_prompts_from_dataset`/prompt 行携带 `images/videos/audios` → 采样器 sample 传 `multi_modal_data`；`RolloutBatch`/`InputFeature` 透传 vision 键（`pixel_values`/`image_grid_thw`，已在 `VLM_CONCAT_FIELDS`）；frozen 辅助（ref/reward/teacher）用同 template 编码多模态；覆盖 GRPO 视觉 / 多模态蒸馏 / 多模态奖励模型。**开放决策**：在线 rollout 的 vision 张量走「训练侧重 encode」（与 infer sidecar 一致，推荐，不持久化 pixel_values）还是「采样器直接回传」。
- **W40 异步生成（async_generate，最后做）**：loop 双缓冲——生成 batch N+1 与训练 batch N 重叠；`async_generate` 从死参（现挂 `_UNWIRED_ROLLOUT_KNOBS`）改为接线。此项设计风险最高。
  - **决策（用户已拍板，源码依据见下）**：
    - **运行时**＝在**统一 `GRPOLoop` 上做 driver 侧双缓冲**，**不**路由到 twinkle 原生 `AsyncMultiLoraGRPOPipeline`（那是多LoRA/TQ/YAML/仅分离式的独立 worker 服务编排，不含 dev 已接的 PRM/路由重放/采样重放/SP/CHORD/teacher，路由过去会劈成两套运行时并使异步 GRPO 特性残缺，违 G3/G4）。
    - **机制**＝`async_generate` 为**布尔**开关，语义＝**1-batch 前瞻**（staleness 固定 ≤1 个 rollout batch），**不设 `max_staleness` 旋钮**。
    - **staleness 上限＝1 的硬约束（源码依据）**：权重同步（`CheckpointEngineManager.sync_weights` 的 NCCL/IPC 路 → vLLM `receive_weights`→`update_weights`→`collective_rpc('update_weights_from_ipc')`→`model_runner.model.load_weights`）**就地改写 live 权重且要求采样器静止**，无法与在飞生成重叠。故 race-free 的 gen∥train 重叠只能是「采样器生成 batch N+1 时 trainer 训练 batch N，收集后（采样器空闲）再推 v_{N+1}」。staleness>1 需 LoRA adapter 版本 pinning（原生 TQ 管线机制，仅 LoRA、全参无解），会造成 LoRA/全参行为分叉（违 G4），本轮不做、不留死旋钮。
    - **off-policy 校正**＝staleness 恒>0，故 `async_generate=True` **强制**要求 `rollout_importance_sampling_mode` 已设（复用 dev 已接的 `rollout_logps` vs 重算 `old_logps` 重要性比，grpo.py:631-634），否则 fail-loudly。
    - **计量单位**＝按 rollout batch 计，与 `num_iterations`（批内重放）正交。
    - **部署门**＝仅 `vllm_mode='disaggregated'`（采样器是独立 Ray actor，异卡才能重叠）；`colocate` 拒绝（单卡靠 `ColocateHandover` enter/exit 分时，物理上无法重叠），fail-loudly。
    - **中立性**＝full-FT 与 LoRA、megatron 与 transformers 一致（生成一律走独立采样器，原则 2）。
  - **实现（三层，依赖序）**：
    - **① twinkle 下沉**：把通用 `GenerationSubmissionMixin`（`submit_generation`/`get_generation_status`/`collect_generation`/`cancel_generation`，纯 `Future` 记账，现居 `twinkle_agentic/async_rl/generation_submissions.py`）下沉到 core `twinkle/sampler/`，async_rl 侧改为从 core 再导出（原生管线 import 不破）。core `vLLMSampler`/`SGLangSampler` 混入该 mixin：core 已有常驻 `self._async_loop`（vllm_sampler.py:78-126），补 `self._generation_submissions={}` + `async def _generate_inputs(...)`。**消重**：把 core `sample` 的异步核抽为 `_generate_inputs`（用 `await self.engine._get_or_load_lora` 而非会死锁的 sync `_load_lora`），sync `sample` 收敛为 `self._run_in_loop(self._generate_inputs(...))` 薄壳；`VLLMSamplerTQ._generate_inputs` 若与 core 等价则删其重复副本改继承。严守 dispatch/collect DP 归属不变量（slice_dp 顶层 list、sampler 侧 collect='flatten'）。
    - **② dev rollout 发行/收集接口**：`RolloutEngine`/`SyncableRollout` 增 `submit_generate(...)->handle` 与 `collect_generate(handle)->List[RolloutSample]`（复用 `generate` 的 trajectory 组装 + `_samples_from_responses`），底层调 twinkle 的 `submit_generation`/`collect_generation`。同步 `generate` 保留（colocate 与非异步路仍用它）。
    - **③ dev `GRPOLoop.fit` 双缓冲**：`async_generate` 时改 1-batch 前瞻调度——`sync_weights(v0)` → `submit_generate(batch0)` → 循环：`collect(batch_N)` → 组 `RolloutBatch`（打分/advantage/PRM/replay 全部复用现路径）→ `submit_generate(batch_{N+1})`（用当前采样器权重 v_N，故 batch_{N+1} staleness=1）→ `train(batch_N)`（与 batch_{N+1} 生成重叠）→ 采样器空闲后 `sync_weights(v_{N+1})`。权重推送只在采样器静止点发生，杜绝竞态。`num_iterations`/mini-batch/CHORD/动态采样等语义不变。
    - **④ config/validate**：`async_generate` 移出 `_UNWIRED_ROLLOUT_KNOBS` 转活旋钮；新增 `_check_async_generate`——`async_generate` 要求 `vllm_mode='disaggregated'`（colocate raise）+ 要求 `rollout_importance_sampling_mode` 已设（否则 raise 指向启用哪种 IS）；`steps_per_generation` 仍 raise 指向 `num_iterations`（不变）。
  - **已知边界（诚实记录）**：staleness 上限恒为 1（就地权重同步的物理约束）；更深的多版本权重隔离/adapter pinning 属 twinkle 后续独立能力，本轮不搭半成品路由、不留死旋钮。
- **W41 训练内 PRM**：`prm`/`prm_weights`/`prm_parallel_spec` 从死参改为接线；PRM 作 frozen 辅助按占卡分档放置（全参独占组，复用 `frozen_auxiliary_distributed_config` + `plan_rl_device_groups` auxiliary_groups）；产出稠密/逐段过程奖励接进 GRPO 的 per-token advantage（复用 `reward.py` 既有 prm 通道 + `compute_reward_model_scores`）。**开放决策（须拍板）**：① PRM 分数 → token 级 advantage 的映射（逐 step 广播 vs 逐 token 插值）；② PRM 过程奖励与 ORM 终局奖励的合并方式（加权 vs 分段）。
- **W42 padding_free + packing 全算法放开**：逐一核实 ppo/rft/opsd/mopd/cpo/orpo/simpo/rm 的 loss 路径在 packed/variable-length 布局下正确（多数经 `forward_only` 的 `unpack_packed_sequences` 归一已成立），核实一个把 `_check_rlhf_padding_free` 白名单放开一个；无法核实的显式 fail-loudly 点名。SP+padding_free 需 flash attn（守卫已存在）。无开放决策（纯核实 + 放开白名单）。

---

## 4. Open decisions（已由用户拍板，全部选 A）

> **决策记录（用户回复"三个都是 A"，五项推荐均为 A，故全部按 A 执行）**：
> - **D1 = A**：`rlhf_type` Literal 新增一等值 `opsd`/`mopd`/`rft`，`run_rlhf.py` 各分派到 `run_opsd`/`run_mopd`/`run_rft`。
> - **D2 = A**：MOPD 用全老师加权聚合，下沉 `twinkle/loss/mopd.py::MOPDLoss(OPSDLoss)` 并注册 `'mopd'`。
> - **D3 = A**：分阶段实现（§9），阶段一先做三新方法 + 直接相关 bug；重构（第 0 块）woven 进各阶段，阶段 4 逐算法收口。
> - **D4 = A**：RFT 默认 `best_of_n`（每 prompt 留最高分），`threshold`/`top_k` 可选。
> - **D5 = A**：OPSD/MOPD 老师 logp 用 sampled-token（response-only）k3 形式，复用 twinkle `OPSDLoss`。
> - **D6（本轮新拍板）**：数据格式主线 = OpenAI 标准 `messages`，直接采用 twinkle 原生 `Trajectory`/`InputFeature`；机制上完全复用 twinkle `sampler` + `Template`（token-in-token-out），legacy 格式入口兼容但主推新格式；embedding/reranker = 多个 `messages` 拼接 + 可选 `label`；OPSD/MOPD 老师用「模型」`forward_only` 而非「sampler」（模型 logits 更稳定）。详见 §1.5 A0、§3.2/§3.3。
>
> 以下保留各选项原文以备追溯；被选中的 A 项即实现依据。

### D6. 数据格式主线与老师表示法（本轮用户拍板）
- **格式主线（选定）**：主推 OpenAI 标准 `messages`，内部一律流转 twinkle 原生 `Trajectory`（原始行）→ `InputFeature`（编码后）；RL/reward 额外字段走 `Trajectory.user_data`（打包 `(key,json)` 对）。机制完全复用 twinkle `sampler`+`Template`：rollout 直接取 `SampledSequence.new_input_feature`，拼接/老师-学生共享 response 用 `concat_input_feature(appended_as=)`，不再手搓 dev 私有的 shift/`completion_mask`/`replace_assistant_response_with_ids`。**legacy 数据格式入口兼容**（既有 format_converter 把 `query`/`response`/`instruction`/`positive_messages` 等归一到 `messages`/`Trajectory`），但文档与 examples 主推新格式。
  - 备选（否决）：继续沿用 dev 现有 `RolloutSample`+私有 shift 约定。否决理由：与 twinkle 原生类型重复、hand-rolled 易错（A2/§2.B 的 off-by-one 类问题正源于此），且 cookbook 已证明原生路成熟。
- **embedding/reranker（选定）**：视为"多个 `messages` 拼接 + 可选 `label` 字段"——anchor/positive/negative 各自 `messages` 编码后拼接，附 float `label`（emb 相似度）或 1/0（reranker 相关性）；沿用 twinkle `InfonceLoss`/`PointwiseRerankerLoss`/`ListwiseRerankerLoss`（§1.5 D/E）。
- **老师表示法（选定 = 模型，非 sampler）**：OPSD/MOPD 老师 = 冻结 twinkle **model** actor，`forward_only` 取 response-only sampled-token logp。
  - 备选（否决）：cookbook GKD 的"第二个 **sampler** + `SamplingParams(max_tokens=0, prompt_logprobs=topk)`"（`rl/gkd/gkd_on_policy.py:279-291`）。否决理由：用户判断**模型自身 logits 更稳定**（sampler 的 prompt_logprobs 受推理引擎数值/kernel 影响），且 model `forward_only` 天然给 sampled-token 形式（匹配 D5-A），无需 top-k 整词表通道。

### D1. 三种新方法在命令面的形态
- **A（推荐）**：`rlhf_type` Literal 新增一等值 `opsd`/`mopd`/`rft`，`run_rlhf.py` 各分派到 `run_opsd`/`run_mopd`/`run_rft`。理由：用户明确"增加几个新训练，都覆盖在 rl 命令里面"，一等值最直观、可发现性最好、与既有 9 值同构；OPSD 虽可借道 gkd，但一等值避免"用 gkd 跑 opsd"的隐式约定。
- **B**：OPSD/MOPD 归入蒸馏家族，复用 `--rlhf_type gkd` + 新增 `--distill_mode {gkd,opsd,mopd}`；RFT 单独一等值。理由：三者都是"学生自采样 + 老师监督"，家族化减少 recipe 数。代价：`gkd` 语义被撑大，CLI 可发现性差。
- **C**：两轴——`rlhf_type`（算法族）× `--rl_paradigm {preference, policy_gradient, on_policy_distill, rejection_sampling}`。理由：正交、可扩展。代价：与 legacy 单轴 `rlhf_type` 契约偏离最大，迁移/兼容成本高。

### D2. MOPD 聚合语义
- **A（推荐）**：`weighted`——每个老师对每条学生轨迹都打分，按 `teacher_weights` 归一加权聚合成一个 token 级蒸馏目标（符合 MOPD 论文"多领域老师融合进一个学生"）。需下沉 `MOPDLoss`。
- **B**：`route`——沿用 legacy 现状，按 `teacher_tag_key` 每样本路由到唯一老师（复用 `route_samples_to_teachers`，无需新 loss，但这其实等价于"分域单老师 OPSD"，不是论文意义的 MOPD）。
- **C**：两者都实现，用 `mopd_aggregation` 开关选。代价：两条路径都要测，体量翻倍。

### D3. 本轮实现范围/排序
- **A（推荐）**：分阶段（§0）——阶段一只做三新方法 + 直接相关 bug（A1），阶段二 rollout 通用化，阶段三死参数/数据管线/辅助模型后端。每阶段单独验收。
- **B**：一次性全做（三新方法 + 全面通用化 + 死参数接线）。代价：单轮不可验证、风险高、与"一步一步"冲突。
- **C**：只做三新方法，通用化与 gap 收口永不排期（不推荐：违背用户"所有训练方法支持所有 backend/sampler/放置"的明确诉求）。

### D4. RFT 的采样-过滤默认策略
- **A（推荐）**：`best_of_n`（每 prompt 保留 reward 最高的 1 条）为默认，`threshold`/`top_k` 可选。理由：RAFT/ReST 主线是 best-of-N，默认最稳、无需调阈值。
- **B**：`threshold` 为默认（`reward >= rft_threshold`）。代价：阈值需按奖励尺度调，默认值难通用。
- **C**：`top_k`（每 prompt 前 k 条）。理由：保留多样性。代价：k 需调。

### D5. OPSD/MOPD 的老师 logp 形式
- **A（推荐）**：sampled-token（response-only）logp，复用 twinkle `OPSDLoss` 的 k3 形式（logits-free，省显存，与既有 `OPSDLoss` 一致）。
- **B**：整词表 top-k logits（复用 GKD 的 `teacher_topk_logprobs`/`teacher_topk_indices` 通道 + `GKDLoss` 的 JSD）。代价：显存/带宽更高，但散度更"全"。论文 headline 是全词表 JSD_beta，A 是其 sampled-token 轻量替代（`OPSDLoss` docstring 已诚实标注此差异）。

---

## 5. Review-mode pass plan（步骤 4 计划，步骤 9 对真实代码执行）

**代码必须成立的不变量：**
1. OPSD/MOPD：学生与老师 forward 的 response token 完全一致（同 id 同序），老师 logp 对齐到学生 loss mask 的 response 位置（`remap_teacher_logps_to_student_frame` 的断言 `s_idx.numel()==t_idx.numel()`）。
2. OPSD 无老师时退化为 0 loss 且过 autograd（`OPSDLoss` 的 `teacher is None` 分支），DDP/FSDP 不见 unused param。
3. MOPD `weighted`（D2-A，唯一实现的聚合）：每个老师是 Ray actor 冻结模型、`forward_only` 打分；权重归一后聚合，老师通道数 == `teacher_model` 数；**单老师时严格退化为 OPSD**（`MOPDLoss(OPSDLoss)` 的核心自洽不变量）。`route` 聚合本轮不实现。
4. RFT：过滤子集确定性可复现（同 seed 同子集）；labels 只在 response 段、next-token shift 与 SFT 一致；空子集 fail-loudly；迭代轮间权重同步确实发生（`sync_weights` 被调用）。
5. `CheckpointEngineManager` 以 `mode=` 构造，disaggregated→`standalone`、colocate→`colocate`（修 A1 后）。
6. 通用化：`rollout_sampler='transformers'` 对 GRPO/PPO/RFT fail-loudly；sglang 的 `sampler_world_size` 计算正确（修 B3）。
7. 死参数：新增字段要么接线、要么 `_validate`/warning 显式点名（不留新死参数）。

**每个变换要推导并实际运行的反例：**
| 变换 | 对抗输入 | 期望（契约） |
|---|---|---|
| teacher_prompt 缺失（OPSD） | 数据列无 `teacher_prompt` | `build_teacher_view()` 返回 False，退化为非 OPSD（或 fail-loudly，按 D1/recipe 约定） |
| 老师/学生 response 长度不等 | 老师 tokenizer 不同 | `remap_*` 断言失败，fail-loudly（"token-in-token-out 需同 tokenizer"） |
| MOPD 权重长度 ≠ 老师数 | `teacher_weights` 少一个 | ValueError（复用 `build_reward_weights` 同款校验风格） |
| MOPD 单老师退化 | `teacher_model` 只 1 个 | `MOPDLoss` 结果 == `OPSDLoss`（自洽不变量） |
| MOPD 老师 Ray actor 打分 | 多老师各占 DeviceGroup | 每老师 `forward_only` 返回 response-only logp，逐位对齐学生 | 
| RFT 全样本被过滤 | threshold 过高 | 空子集 fail-loudly，不静默 0 步 |
| RFT best_of_n 并列 | 多条同 reward | 确定性 tie-break（按索引），可复现 |
| colocate 采样器 > 训练卡 | `sampler_world_size > nproc_per_node` | `plan_rl_device_groups` ValueError（既有） |
| transformers 做独立 rollout | `rollout_sampler='transformers'` + grpo | fail-loudly 指向 vllm/sglang |

**复用/架构/分层/范围审计：** 三新方法是否最大化复用既有 recipe/reward/loss（不重写）；`MOPDLoss` 是否该下沉 twinkle（是，通用 loss）；多老师路由是否新增跨栈 import（阶段一尽量不，阶段三统一下沉）；是否有超需求抽象（剃刀）。

**无 oracle 的一侧（严谨度要往这边压）：** MOPD `weighted` 聚合的数值正确性没有现成参考实现（论文无官方 code 在本仓），需用"退化为单老师时 == OPSD"的自洽检查 + 手工小算例锚定；多老师 Ray actor 的 `forward_only` per-token logp 与 legacy HTTP 老师 logp 的等价性无直接 oracle，需用同模型同 tokenizer 下两条路径（若 legacy 仍可跑）或手工前向对齐钉住；RFT 过滤后 SFT 的 loss 缩放与 legacy best-of-n 的一致性无直接 oracle，需与 legacy 行为对比钉住有意差异。

---

## 6. Testcase-planning sketch（步骤 5 初步草图，步骤 10 对真实代码定稿并执行；用户要 UT 前不写测试）

| # | case | input | expected（契约） | kills |
|---|---|---|---|---|
| 1 | OPSD 单老师自蒸馏（disable_lora） | LoRA 学生 + `teacher_prompt` 列 | 学生/老师 response token 一致，loss 有限且下降，无 unused-param | 老师视图构造/对齐错 |
| 2 | OPSD 无 teacher_prompt | 数据列缺失 | 退化/fail-loudly（按约定） | 静默走错分支 |
| 3 | OPSD 无老师退化 | teacher=None | 0 loss 过 autograd | DDP/FSDP 崩 |
| 4 | MOPD weighted 两老师 | 2 本地老师 + weights | 加权聚合 == 手算；单老师时 == OPSD | 聚合/归一错 |
| 5 | MOPD 多老师 Ray actor | 2 冻结老师各占 DeviceGroup | 各 `forward_only` 打分，加权聚合，无 HTTP | 老师放置/打分路径错 |
| 6 | MOPD 权重长度不符 | weights 少一个 | ValueError | 静默错配 |
| 7 | RFT best_of_n | N=4 采样 + accuracy 奖励 | 每 prompt 留最高分，SFT labels 只 response 段 | 过滤/label shift 错 |
| 8 | RFT threshold 全过滤 | 阈值过高 | fail-loudly | 静默 0 步 |
| 9 | RFT 迭代权重同步 | iterations=2 | 第 2 轮采样器权重 == 第 1 轮更新后 | 未同步（伪 on-policy） |
| 10 | CheckpointEngineManager mode | disaggregated / colocate | 分别 `standalone`/`colocate`，无 TypeError | A1 回归 |
| 11 | rollout_sampler=transformers + grpo | 不可行组合 | fail-loudly 指向 vllm/sglang | 静默降级 |
| 12 | sglang sampler_world_size | tp*pp*dp | `plan_rl_device_groups` 设备组正确 | B3 回归 |

端到端原则（reference.md 教训）：用真实模板（只加载 tokenizer、`load_model=False`）+ 脚本化采样器（返回预置 token）+ 假 reward，无 GPU 也能驱动真实引擎跑完整流程；慢测（`@pytest.mark.slow`+`accel`）真起 Ray+vLLM/SGLang 打真请求。反向验证：去掉正确行为测试须变红。

---

## 7. Verification plan

- **写框架代码阶段**：仅 AST 校验（`ast.parse` 全部改动文件，命令模板见 `dev-module-authoring/reference.md`）+ 静态核对运行期契约（twinkle `extra='forbid'` schema 字段名、`torch_loss_mapping` 注册键、导出符号、import 路径、CLI flag→映射函数）。
- **真实改动路径**：三新方法各跑一次最小 e2e（Qwen2.5-0.5B，单卡/双卡），驱动真实 recipe→rollout→loss→optimizer→save，读回 checkpoint（reload 写出的文件，不信内存值）。
- **数值**：OPSD/MOPD 的 loss 与"单老师退化 == OPSD"自洽检查；RFT 过滤子集与手工打分一致；容差与依据（bf16）在补测阶段写明。
- **通用化**：vllm 与 sglang 各跑一次 GRPO/RFT 最小 e2e（≥2 卡），colocate 与 disaggregated 各一次；transformers 独立 rollout 的 fail-loudly 用单测钉住。
- **解释器/环境**：永远 `/usr/local/bin/python -m pytest ... --import-mode=importlib`；`MODELSCOPE_CACHE=/mnt/workspace/.cache/modelscope/hub`、`VLLM_USE_MODELSCOPE=True`；不设 `HF_HUB_OFFLINE`；`MASTER_PORT=$((29000 + ${GPUS%%,*}))`；后台任务加 wait/SIGHUP 保护；输出过滤 PAI DSW banner。
- **examples 冒烟**：每新方法在 `examples/v5/rl/` 下加 `.sh`（+ 可选 client），真跑 `swift rl ...` 验收（用户验收线：直接运行 examples 可用）。

---

## 8. Definition of Done

**总体 DoD（用户一等目标：所有 RL 算法准确、可用、代码优良）：**
- [ ] §2.0 审计矩阵 12 种算法 × 7 维度全部收口为 ✓ 或 —（无 ✗ 遗留），每格有验收依据。
- [ ] 每种算法都有一条真能跑通的 `swift rl ...` 命令（examples 冒烟）并产出正确 checkpoint（读回验证）。
- [ ] 无死参数（每个声明旋钮要么生效、要么 fail-loudly/warning 点名）；无静默降级；不可行组合 fail-loudly。
- [ ] 代码质量审查通过：分层正确（cli/config/recipe/builders 四层镜像）、最大化复用、通用能力下沉 twinkle、无跨业务/跨栈乱引用、剃刀（无超需求抽象/半成品路由）、命名与注释教科书化。

**Feature DoD（称"已实现"，暂不含测试）：**
- [ ] 本 plan written 且经用户确认后才动代码。
- [ ] 三新方法（OPSD/MOPD/RFT）按 D1–D5（全 A）实现；写代码阶段 AST 校验通过。
- [ ] A1（`CheckpointEngineManager(mode=)`）修复；RFT 复用 rollout 不再崩。
- [ ] review-mode pass 对真实代码执行，§5 每个反例实际运行且通过。
- [ ] 无未论证的框架改动；开放决策 D1–D5 已拍板（全 A），未静默默认。
- [ ] 新增字段无"死参数"（接线或 fail-loudly/warning 二选一）。
- [ ] 通用化接缝按阶段落地；不可行组合 fail-loudly（不静默降级）。
- [ ] 局限诚实记录（E 类未实现项、辅助模型后端、多模态等），不搭半成品掩盖。

**Test DoD（用户要 UT 时才适用）：**
- [ ] §6 PLAN 表定稿，覆盖每个适用 corner-case 类别或 N/A-with-reason。
- [ ] 测试端到端（真实集成链路，非桩孤立单方法）、完整、full；每条反向验证（去掉修复变红、恢复变绿）。
- [ ] 真实路径验证；checkpoint 读回；数值在声明容差内。
- [ ] 先修 G 类既有破损测试，再跑整体门禁（fast+slow 同进程顺序）。

---

## 9. 分阶段实施顺序（D3-A 已拍板；重构 woven 进各阶段）

**阶段 0（审计，纯读不改）**
1. 对既有 9 种算法逐一过 §2.0 审计矩阵 7 维度，把"待审"替换为 ✓/✗/— + 一句缺陷说明。
2. 把 §2 的 A–G 缺陷映射到具体算法/文件/行，形成可执行重构工单清单。
3. 产出后让用户确认阶段一起点（不改代码，只交审计报告）。

**阶段一（三新方法 + 直接相关 bug）**
1. 修 **W00/A1**（`run_grpo.py` `CheckpointEngineManager(mode=)`）+ **W02/A2**（`completion_mask` off-by-one，RFT 复用前必修）+ **W01/A3**（离线偏好 `chosen/rejected_labels` 补 next-token shift，5 算法静默训错）。
1b. **类层次抽取（§3.8，行为等价）**：抽 `TrainLoop` 主干（step/tracker/checkpoint/gc/resume/callback）+ `OnPolicyLoop`/`DistillLoop` 两级基类；把 `SFTLoop`/`GRPOLoop`/`GKDLoop` 收敛为子类（只搬脚手架、不改行为）；引入 `RolloutBatch` 值对象与 `Rollout`/`Teacher` 协议（`FrozenModelTeacher`/`DisableAdapterTeacher`/`MultiTeacher`，解 **W29** 老师抽象非多态）。抽完跑既有 e2e 须与抽前一致（review-mode 核对）。`PreferenceLoop`/`PPOLoop` 收敛留阶段三/四。
1c. **蒸馏对齐 GRPO on-policy 结构（W33+W34 合并，行为变更，排在 1b 之后）**：把 design B 同构应用到 gkd/opsd/mopd——CLI 将蒸馏纳入 `mode='ray'` 翻转**并设 `use_vllm`/`vllm_mode`（rollout 走独立采样器，与 grpo/ppo/rft 同；用户拍板：生成一律 vllm/sglang，不再就地）**；recipe 算 `dp_size = build_ray_dp_mesh(distributed_config).data_world_size`、`batch_size = per_device_train_batch_size × dp_size`，学生 rollout 经 `SamplerRollout`（采样器 DP）、`forward_only`/`forward_backward` 喂全局批由 slice_dp 切分；冻结老师按 §3.3 建 Ray actor（`mode='ray'`+`remote_group`，`plan_rl_device_groups` 扩为 trainer+N teacher 组），`disable_lora`/dynamic-self 老师在 DP 学生上 `forward_only`。验收：蒸馏 DP=1+DP=2 双复核（§3.8(6)）。详见 §2.0 W33。**ray 化连带核对（易漏）**：蒸馏移到 `mode='ray'` 后会走 GRPO 踩过的 ray 路径，须逐一确认对蒸馏也成立——Bug#4（全局 DeviceMesh：蒸馏走 `TrainAssembly.initialize_twinkle` 而非 `run_grpo._initialize_twinkle_rl`，须核实 ray 分支装了全局 mesh，否则 `device_mesh=None`+`remote_group` 的老师/学生被拒）、Bug#5（driver 不读 model proxy 的 attr、改调 `@remote_function` method；`do_grad_sync` 用 loop 自己的 `micro_step` 复刻）、Bug#7（`forward_only` `lazy_collect=False` 已在 twinkle 全局修，蒸馏自动受益，但须核实老师 `forward_only` 返回物化 dict 而非惰性句柄）。
1d. **〔已并入 1c〕** 生成后端通用不再是独立步骤：1c 的采样器 rollout 已使蒸馏两后端皆通（`TransformersModel`/`MegatronModel` 都混入 `CheckpointEngineMixin`，复用 design B 的 `SamplerRollout`/`SyncableRollout`/`plan_rl_device_groups`）。就地 `model.generate` 只保留给训练中间 eval（transformers-only；有采样器时 eval 复用采样器，megatron 亦可）。`validate.py` 的 fail-loudly 相应改为：无采样器场景下 `megatron × 就地 eval generate` 拒绝。验收：蒸馏在 transformers 与 megatron 两后端各跑通 DP=1+DP=2（并入 1c 双复核）。详见 §2.0 W33/W34。
2. 配置层〔已完成〕：`rlhf_config.py` 加 OPSD（`opsd_reverse`）/MOPD（`teacher_model` 拓宽为 `Union[str,List[str]]`、`teacher_weights`、`teacher_parallel_spec`）/RFT（`rft_num_samples`/`rft_select`(默认 best_of_n)/`rft_threshold`/`rft_top_k`/`rft_iterations`/`rft_max_samples_per_prompt`）字段（带 `#:` 注释）；`rlhf_type` 扩值 `opsd`/`mopd`/`rft`（D1-A）；`rollout_config.py` 加 `rollout_sampler`（`validate.py::_check_rollout_sampler` 对非 vllm fail-loudly 指向阶段二）；拆 **W16**（`temperature`=蒸馏/打分温度 vs 新 `sampling_temperature`=采样温度，`run_gkd._gkd_sampling_params` 已改用后者、None 回退保持旧行为）。附带修正：`process.py::_RLHF_BETA_DEFAULTS` 补 `opsd/mopd/rft: 0.0`（否则落入 0.1 兜底、给自蒸馏/RFT 错误加上 ref-KL）；`loss/configure.py::_RLHF_LOSS_NAME` + `_online_loss_kwargs` 注册 `opsd`/`mopd`（forward `reverse`/`beta`，温度走 forward 不走 loss）。**核对结论**：新字段名在各 config 间唯一，CLI `_merged_parse_class` 自动归属，无需 `field_owners` 显式 pin；`legacy_coverage` 无需改（新 flag 非 legacy-only）。
3. 下沉 twinkle〔已完成〕：新增 `twinkle/loss/mopd.py::MOPDLoss(OPSDLoss)`（D2-A 全老师加权、概率混合 logsumexp）+ 注册 `'mopd'`；`opsd.py` 抽出 `_student_logps_and_mask`/`_per_token_distill_loss` 供复用（OPSD `__call__` 逐字节等价）。详见 §3.3。
4. recipe〔已完成，最小脚手架〕：`run_opsd.py`（D5-A sampled-token 老师 logp、D6 老师=模型 `forward_only`）、`run_mopd.py`、`run_rft.py`（D4-A best_of_n 默认）；`run_rlhf.py` 分派（`opsd`/`mopd` 不传 `rollout_config`，`rft` 与 `grpo` 同形）。**与计划的偏差（先 b 再 a）**：1b 类层次抽取推迟，故此步用最小脚手架直接继承既有 loop —— `OPSDLoop(GKDLoop)`（只覆盖 `_teacher_kwargs`/`_forward_kwargs`）、`MOPDLoop(OPSDLoop)`（只覆盖 `_teacher_kwargs` 为 K 老师 channels + `teacher_weights`）、`RFTLoop(GRPOLoop)`（覆盖 `fit` 为"选择→cross_entropy SFT"，loss 走 `configure_loss` 非 `_RLHF_LOSS_NAME`）。新增共享 `swift/dev/recipe/_distill.py`（`response_positions`/`response_ids_from_feature`/`response_logps`/`teacher_view_messages`/`build_privileged_teacher_feature`）作为新蒸馏 recipe 的唯一老师视图原语；`grpo.py` 内 RLSD/SDAR 的重复老师 helper **本步不动**，收敛到 `_distill.py` 留 1b（避免把爆炸半径扩到 GRPO）。数据面：老师视图用 `teacher_prompt` 替换**首条** user 消息 + 共享 response token（dev grpo 约定；rl_core legacy 替换末条，单轮等价）。**OPSD/MOPD v1 不施加蒸馏温度、无 reference-KL**（`OPSDLoss.__call__` 既不读 `beta` 也不读 `temperature`），故 `_online_loss_kwargs` 只 forward `reverse`，`beta`/`temperature` 在 validate 侧显式拒绝而非静默丢弃。静态契约核对通过（导入符号/`RLHFConfig` 字段名/`GRPOLoop`·`GKDLoop` 构造 kwarg/loss 注册键/`MOPDLoss._is_multi_teacher` 对 K×N channels 的判别全部解析）；修 1 个真实 bug：`run_mopd._build_mopd_teachers` 曾把 `None` 传给 `configure_frozen_adapter(adapters=)`（其内 `len(adapters)` 必 TypeError），改为 `[]`。
5. CLI〔已完成，但含 W33 放置缺陷待修〕：`rlhf.py` 的 Ray+vLLM 强制放置集合从 `{grpo,ppo}` 扩为 `{grpo,ppo,rft}`（opsd/mopd/gkd 进程内 `mode='local'` 生成，**不可**翻转）；**〔W33+W34 合并修订，用户拍板〕**「不可翻转」是错的——蒸馏须纳入 `mode='ray'` 翻转**并设 `use_vllm`/`vllm_mode`**（rollout 一律走独立采样器 vllm/sglang，与 grpo/ppo/rft 同；不再就地生成），详见 §2.0 W33/W34 与 §3.5；就地 `model.generate` 只留给训练中间 eval（transformers-only），`validate.py` 相应加「无采样器场景 `megatron × 就地 eval generate`」前置 fail-loudly；新字段名各 config 间唯一，`field_owners` 无需新增；`legacy_coverage` 无需改（新 flag 非 legacy-only）。validate 侧：`_check_distillation` 覆盖 gkd/opsd/mopd（opsd/mopd 拒绝非零 `beta`）；`reward_template` 允许 grpo+rft；`ignore_data_skip` 拒绝集补 opsd/mopd/rft；padding_free/packing 白名单仍 `{grpo,dpo,kto,gkd}`（opsd/mopd/rft 显式拒绝，不静默）。
6. examples〔已完成〕：`examples/v5/rl/{opsd,mopd,rft}/*.sh` + `legacy_alias.sh` + `opsd/opsd_plugin.py`，数据 `data/{opsd,prompts,rft_math,legacy_query}.jsonl`（主推 OpenAI `messages`/`Trajectory`；legacy 别名另留兼容示例）。4 条示例在 parse 与 process+validate 两层均通过。期间修 1 个真实 bug（Bug#2）：`RLHFConfig.teacher_model` 原为 `Union[str,List[str],None]`，被 `parser._patch_type_hints` 特判成单值 `Optional[str]`（`str_list_none` 分支），致 MOPD 的 K 个教师无法从 CLI 表达（`--teacher_model A B` 解析失败）；改为真 `Optional[List[str]]`（对齐 `reward_model` 的多值模式，不动共享 parser），并抽出 `assembly.single_teacher_id()` 统一 GKD/OPSD/GRPO-RLSD 三处「只允许一个教师」规则；`process._derive_rlhf_teacher` 归一化元组补 `teacher_model`、自蒸馏判据改 `[model_config.model]` 比较。
7. 全改动文件 AST + 静态契约核对〔已完成〕；最小 e2e 冒烟〔部分完成，见下〕。用完整训练环境 `/usr/local/bin/python`（3.12，torch2.11/twinkle0.4/vllm0.23/ray2.56/math_verify 齐备；注意裸 `swift` 命中的 miniconda3.14 缺 ray/math_verify，冒烟须显式 `/usr/local/bin/swift`），Qwen2.5-0.5B，`max_steps 2` 缩短，输出 /tmp，跑完读回 checkpoint：
   - **OPSD**（单卡 mode=local，动态自蒸馏 teacher=None；**W33：仅 DP=1、须扩 DP>1，非验收态**）：**通过**。recipe→generate→teacher forward_only→OPSDLoss→optimizer→save 全跑通，loss 0.1104→0.1082，checkpoint-2/final 落盘，adapter 读回 336 张量/4.4M 参数/全 finite。
   - **MOPD**（单卡 mode=local，两教师 `teacher_weights 0.6 0.4`；**W33：仅 DP=1、须扩 DP>1，非验收态**）：**通过**，但先修 1 个通用真实 bug（Bug#3）：辅助模型（teacher/ref/reward/value）在 recipe 内以裸 hub id `ModelConfig(model=<id>)`+`build_model` 构建，**不经** `process._resolve_model`（它只解析主模型 id→本地 snapshot 目录），故 `build_config` 拿裸 id 调 transformers `from_pretrained` 直接 `OSError: Can't load the configuration`。根因修法（单一共享接缝）：在 `builders/model.py::build_model` 顶部对 `model_config.model` 调 `safe_snapshot_download`（对已解析的本地目录短路，主模型是廉价 no-op；仅裸 id 辅助模型真下载），一处覆盖 GKD/OPSD-独立教师/MOPD/GRPO-ref+teacher/PPO/reward。修后 loss 0.0823→0.0148，checkpoint 读回全 finite。**已知局限**：辅助模型解析走 `USE_HF` env 默认、不继承主路径的 `use_hf`/`hub_token`（gated 辅助模型需另行 threading，未做）。
   - **RFT**（Ray+vLLM colocate，accuracy ORM，DP=1）：**通过**（`ga=1`、`rft_iterations=2`、`max_steps=3`：3 个真优化器步 loss 0.4461→0.1557→0.3211，`cur_step=3`/`consumed_train_samples=3`，checkpoint 读回 336 张量全 finite）。期间修 3 个通用真实 bug：
     - **Bug#4（ray 无全局 DeviceMesh）**：twinkle ray 模式（不同于 local）**不**自动安装全局默认 `DeviceMesh`（`initialize` 仅 local 分支建），故任何 `device_mesh=None`+`remote_group` 的 ray 对象被拒（`Set device_mesh=DeviceMesh(...) to enable ray`）；训练模型靠 `_apply_ray_placement` 显式传 `build_ray_dp_mesh` 规避，但 rollout sampler 以 `device_mesh=None` 构建、依赖全局回退。根因修法：`run_grpo._initialize_twinkle_rl` 给 `twinkle.initialize` 传 `global_device_mesh=build_ray_dp_mesh(distributed_config)`（与模型同一 pure-DP mesh，正是 colocate sampler 契约要求共享的 mesh），一处覆盖 GRPO/PPO/RFT。
     - **Bug#5（driver 读 ray proxy 的 attr）**：`grpo.py::_active_group()` 在 **driver** 上读 `self.model.optimizer_group[...]` 并调 `do_grad_sync`/`calculate_metrics`，但 Ray 下 driver 侧 model 是 PROXY、`optimizer_group` 住在 worker → `AttributeError`。**修法遵循「调 method 不读 attr」（无需改 twinkle）**：删除 `_active_group`，改用 model 上**已存在**的 `@remote_function calculate_metric(is_training=True)`（在 worker 上读 optimizer_group）；`do_grad_sync` 是纯函数（无 all-reduce、不自增），在 driver 侧用 loop 自己的 `micro_step` 复刻为 `train_loop.is_grad_sync_boundary(micro_step, ga)`（SFTLoop 早已如此，GRPO/RFT 收敛到同一 helper）。与 twinkle 逐位等价已核对（transformers.py:1096-1102 forward_backward 内自增 cur_step 后调 do_grad_sync，loop 的 micro_step 与之同步）。
     - **Bug#6（DP>1 每样本 batch 太小，未修，见下）**。
   - **GRPO**（Ray+vLLM colocate，accuracy ORM，DP=1，LoRA+`beta>0`→`disable_lora` 参考）：**通过**（`ga=1`、`max_steps=2`：2 个优化器步 loss 0.0000——GRPO 首个内迭代 ratio=1、组内归一化优势均值 0，故 loss≈0 属正确；`cur_step=2`/`consumed_train_samples=2`，checkpoint 读回 336 张量全 finite）。此跑专门验证 `GRPOLoop.fit`（`RFTLoop` 覆盖 `fit`，RFT 冒烟不经 grpo.py 的 fit）与 Bug#5 修法，并暴露+修复 **Bug#7**：
     - **Bug#7（ray driver 上 `forward_only` 返回惰性句柄而非 dict）**：`forward_only` 装饰为 `@remote_function(dispatch='slice_dp', collect=collect_tensor_dict)`，`lazy_collect` 缺省 None→跟随全局 `_lazy_collect=True`；ray driver 分支（infra:1132-1149）因此返回**未 collect 的可调用句柄**，而非物化 dict。`_response_logps` 判 `isinstance(out, dict)` 为假→`logps=None`→`RuntimeError: reference/teacher forward returned no logps`。**这是系统性 bug**：GRPO 参考(`disable_lora`走 ray policy proxy)/GRPO 教师(当为 policy)/PPO value·ref·reward(`run_ppo.py:357/371/381/383`) 全部把 `forward_only` 返回值当 dict 用，ray 下**都**会中招；OPSD/MOPD/DPO/GKD 因 `mode='local'`（infra:1045-1046 直接执行 func、根本不走 lazy 分支）而幸免——这正解释了「local 蒸馏通过、ray GRPO 失败」。**根因修法（下沉 twinkle，一处覆盖全部 ray 消费者）**：`forward_only` 的语义就是「把 outputs 交回调用方」，其返回值**总被消费**、从无 fire-and-forget 用法，故正确装饰是 `lazy_collect=False`（与同样返回被消费值的 `calculate_metric` 一致；`forward_backward`/`forward` 保持惰性=训练步 fire-and-forget）。改 `transformers.py:749` 与 `megatron.py:430`（megatron 有同一潜在 bug，`sync=True` 只定同步派发、不物化返回）。**爆炸半径已核**：server 的 `TransformersModel.forward_only` 自带独立装饰（transformers_model.py:103），不受基类改动影响；local 模式不 consult lazy_collect，OPSD/MOPD/DPO/GKD 与 HTTP server 可证不受影响。修后 GRPO 参考路径 `disable_lora` forward 正常物化、0 次 "returned no logps"。
   - **Bug#6（loop 逐样本喂 `forward_backward`，应改为按 batch 喂——真设计缺陷，非「DP 限制」）**：先前误判为「DP>1 限制、推迟到 1b」，**用户纠正**：twinkle cookbook 的 GRPO（`cookbook/client/twinkle/short_math_grpo.py:273-280`）把**整个 rollout batch 一次喂入**——`forward_backward(inputs=all_input_data, advantages=advantages, old_logps=all_old_logps)`（`advantages`/`old_logps` 是与 batch 平行的 list），随后一次 `clip_grad_and_step()`，`step` **每个 rollout batch 自增一次**。dev 的 `grpo.py::fit`（607-632）与 `run_rft.py::fit` 却 `for item in batch:` **逐样本**调 `forward_backward(inputs=[one_sample], ...)`——这既不符合 cookbook 契约，也正是 DP>1 报 `Batch too small` 的**根因**：`slice_dp`（infra:698-718）把 `inputs` 列表切给 `dp_size` 个 worker，逐样本时 `len(inputs)=1 < dp_size` → 部分 rank 空数据。**按 batch 喂即天然修好 DP**（slice_dp 正常切分；SFTLoop 本就按 dataloader 整批喂，故无此问题）。**修法（设计 B：忠实 mini-batch GRPO，用户拍板「B 显然更好」）**——不是把整个 rollout 一次喂入（那是 cookbook 的最小 demo，忽略 `per_device_train_batch_size`），而是切成 mini-batch：
     - **mini-batch 宽度** `train_batch_size = per_device_train_batch_size × dp_size`，与 SFT 的 `_twinkle_loader_layout` 全局批宽同源（`builders/dataset.py:325`）。ray 下 `dp_size = build_ray_dp_mesh(distributed_config).data_world_size`（= nproc_per_node，pure-DP）。由 **recipe**（run_grpo/run_rft，二者都持有 distributed_config + train_config）算好，作为**单个整数** `train_batch_size` 传入 loop（loop 不懂分布式；RFTLoop 经 `*args/**kwargs` 透明转发）。
     - **`grpo.py::fit`**：每 rollout 产出的样本列表按 `train_batch_size` 切成等长 mini-batch；**尾部不足一个 mini-batch 的余数丢弃**并 warn 一次（保证每次 `forward_backward` 恰好 `train_batch_size` 行 → slice_dp 每 rank 恰好 `per_device_train_batch_size` 行，均匀、不触发 Batch too small；对齐 Megatron「global_batch 精确不变量」哲学，dataset.py:416）。每 mini-batch 组装**平行 list**（`inputs`/`advantages`/`old_logps`/`rollout_logps`/`ref_logps`/`teacher_logps`/`truncated` 各一条，长度 = mini-batch 大小）**一次** `forward_backward`；`micro_step += 1` 改为**每 mini-batch**（不再每样本）。`ga` 跨 mini-batch 累积、`clip_grad_and_step` 每 mini-batch 调、boundary 仍由 `is_grad_sync_boundary(micro_step, ga)` 判（语义不变）。`num_iterations` = 对同一 rollout 重复的 epoch 数（每 epoch 重切 mini-batch）；`max_steps` = 优化器步数（global_step），语义不变。
     - **`run_rft.py::fit`**：同样把 `_select_samples` 选中的样本按 `train_batch_size` 切 mini-batch，每 mini-batch 一次 `forward_backward(inputs=[s.input_feature for s in mb], gradient_accumulation_steps=ga)`（RFT 是纯 SFT，无 advantages/old_logps）。
     - **CHORD 局限（fail-loudly，不退化成静默训错）**：chord 行追加在 `inputs` 尾部、`chord_count` 是广播标量、slice_dp 连续切分 → chord 行只落到最后一个 rank，dp>1 下各 rank `rl_count = batch - chord_count` 错位（会把真 policy 行当 chord）。这是**既有**设计限制（旧逐样本喂法在 dp>1 直接 crash，从未支持 chord+dp>1）；为不把「loud crash」退化成「silent wrong」，在 run_grpo 构建 loop **前**校验：chord 激活且 dp_size>1 → raise，指向 dp=1 或去掉 chord。dp=1 下 chord 与改前逐位一致（`_chord_mu`/`_next_chord_features` 每 mini-batch 调一次，count = `chord_sft_per_device_train_batch_size or 1`）。
     - **示例步数预算需重定〔已完成〕**：旧逐样本喂法**忽略** `per_device_train_batch_size`（每 forward_backward 恒 1 样本），design B 首次真正 honor 它，故 `rft.sh` 的 `max_steps`/`ga` 按新语义重算：数据 4 prompts × `rft_num_samples 4` = 16 rolled out/round，best_of_n 每 prompt 恰留 1 → **4 selected/round**（best_of_n 确定，非 reward 依赖）；`train_batch_size = per_device 1 × dp 2 = 2` → **2 mini-batch/round**；`rft_iterations 3` → 6 mini-batch 总。**关键坑（先误设 ga=2 后纠正）**：优化器步边界走 twinkle「晚一步」规则 `is_grad_sync_boundary = ga==1 or ((micro_step-1)%ga==0 and micro_step>1)`（syncs at micro_step=ga+1,2ga+1,…），故 6 mini-batch 下 `ga=2` 只在 micro_step 3、5 触发 = **2 步**、且尾部窗口（micro_step 5-6 之后的 flush 需 micro_step=7 永不到）**被丢**、`max_steps` 成死参数。最终定 **`ga=1`**（每 mini-batch 一步 = **6 优化器步**，无尾窗丢弃）、`max_steps 12→6`、`save_steps 12→6`——恰与已通过 DP=2 RFT 冒烟（step 1-6）同形。`.sh` 头注「Step budget」段写明该算式 + 「晚一步」规则 + 为何 ga=1。
   - **DP=2 冒烟（nproc_per_node=2，验收线「直接运行 examples 可用」）〔已完成〕**：RFT 与 GRPO 均在 2 卡 colocate 跑通、checkpoint 读回全 finite。期间暴露+修复 2 个通用真实 bug（皆 DP>1 或 round≥2 才触发，DP=1 冒烟未覆盖）：
     - **Bug#8（colocate 第二次 `wake_up()` 被 vLLM 早退，KV cache 永不唤醒）**：`SamplerRollout.sync_weights` 与 `assembly.eval_enter` 的显存调度先 `wake_up(tags=['weights'])` 同步权重、offload trainer，再 `wake_up()` 唤醒 KV cache 供 generate。但 `wake_up()` 的 tags=None 展开为 `['weights','kv_cache']`，而 vLLM（`vllm/v1/executor/abstract.py:318-355`）在**首个不在 `sleeping_tags` 的 tag 上 `return` 整个唤醒**——`'weights'` 已被上一步唤醒、不在 sleeping_tags，故第二次 `wake_up()` 命中早退、**KV cache 保持 discarded**，随后 generate 在已释放显存上建 attention metadata → `CUDA error: invalid argument`（flash_attn.py:547）。**round 1 掩盖此 bug**：新建 sampler 尚未 sleep，两次 wake 皆 no-op；仅 round≥2（sleep 之后）才炸，故 DP=1 单 round 冒烟通过、DP=2 多 round 立崩。**根因修法**：第二次唤醒**只**要 `wake_up(tags=['kv_cache'])`（weights 已醒），两 wake tag-不相交。改 `run_grpo.py::SamplerRollout.sync_weights` + `assembly.py::eval_enter` 两处（colocate 交接逻辑此二处重复，抽公共留 1b）。`ARGUMENTS_MIGRATION.md:535` 同步更新调度文字。修后 DP=2 RFT 3 round 跑完、每 round sleep 释放 ~68GiB 再正常 wake。
     - **Bug#9（打分侧逐样本 `forward_only`，DP>1 `Batch too small`——Bug#6 的打分同胞）**：`grpo.py::_response_logps` 原**逐 feature** 调 `forward_only(inputs=[one])`；`forward_only` 是 slice_dp 方法，长度 1 的 inputs 使 dp>1 下非首 rank 无数据 → `ValueError: Batch too small for 2 workers`。这是 Bug#6（训练侧逐样本 `forward_backward`）在**打分侧**（reference/teacher/rollout-offpolicy old_logps 重算）的同构缺陷，DP=1 单 rank 时不触发，故此前 GRPO DP=1 冒烟通过、DP=2 立崩。**根因修法**：`_response_logps` 改**按 `train_batch_size` 分块** `forward_only(inputs=chunk)`（`_score_chunks` 把不足一块的尾折进最后一块，保证每块 ≥ dp_size 行不饿死 rank）；`forward_only` 返回**全序列右填充** `logps [n, seq_len]`、按提交序（`collect_tensor_dict` pad+stack），故每行按该 feature 自己的 response positions 索引取对数概率、右填充不移位（`_row_response_logps` 兼容「response-only」与「全序列填充」两形态）。3 处 caller（`_reference_logps`/`_teacher_logps`/old_logps 重算）改传 feature **列表**。修后 GRPO DP=2（LoRA+`beta 0.001` 走 `disable_lora` 参考打分）step 1-4 loss -0.4979/0.4918/-0.4965/0.4961（负/振荡属正确：首个内迭代 ratio=1、组内归一化优势可负），无 `Batch too small`、无 CUDA 崩，checkpoint-final 读回 336 张量/4.4M 参数全 finite、168/168 `lora_B` 非零（`lora_B` 初始为 0，全非零证明 4 个优化器步真训练了 adapter）。
   - **DP=2 冒烟结论**：design B（mini-batch 宽度 `per_device × dp_size`、按 mini-batch 喂 `forward_backward`/`forward_only`、`ga` 跨 mini-batch 累积）在 RFT（`RFTLoop.fit`）与 GRPO（`GRPOLoop.fit`/`_mini_batch_kwargs` + `disable_lora` 参考打分路径）两条 loop 上于 DP=2 端到端验证通过。**已知重复（留 1b）**：colocate 显存交接逻辑在 `run_grpo.py::SamplerRollout` 与 `assembly.py::eval_enter/eval_exit` 两处重复（Bug#8 须双改即为此），类层次抽取时收敛为单一 `Rollout` 交接原语。
   - **未决（需用户拍板，不擅自扩范围）**：`examples/v5/rl/` 只有 opsd/mopd/rft 三条示例（step 6 明确只覆盖 3 个新方法），**无 GRPO 示例**。GRPO 是旗舰 on-policy 算法、属「12 算法皆可用+示例可直接运行」总目标，但其示例更贴近阶段三/四「逐算法收口」而非阶段一。是否本步补一条 `examples/v5/rl/grpo/grpo.sh`（与刚验证的 DP=2 GRPO 冒烟同形），或留到阶段四，请用户定夺。

**阶段二（rollout 采样器通用化）**
vllm/sglang 接缝（`run_grpo.py` backend 可选）+ `plan_rl_device_groups` sglang world size（修 B3）+ 权重同步 mode 校正 + transformers 独立 rollout 的 fail-loudly。

**阶段三（gap 收口 + 重构）**
死参数接线（C 类逐项）、数据管线缺陷（B1/B2）、跨栈 legacy import 下沉（D 类）、"server"词汇清理（H 类：移除 `vllm_server_*`/`teacher_model_server`/`_Remote*Teacher` HTTP 分支）、复用未用上的 twinkle 能力（F 类）、辅助模型（ref/teacher/reward）后端通用化。

**阶段四（收口验收）**
修复既有破损测试（G 类）+ 每算法逐一 e2e 验收 + 代码质量审查（分层/复用/剃刀/命名/注释）+ §2.0 矩阵 12×7 全格收口，确保 12 种算法全部准确、可用、代码优良。

---

### CONFIRMATION GATE
D1–D6 已全部拍板（D1–D5 = A，D6 见下），并已把"梳理与重构现有 dev RL 代码、使所有 RL 算法准确可用且代码优良"提升为一等目标（§0 第 0 块、§2.0 审计矩阵、§8 总体 DoD、§9 阶段 0/阶段 4）。

已按用户三轮补充更新：
1. **无-server / 全 Ray 架构前提**（文首架构前提、§1.2、§2.C/§2.H、§3.1/§3.2/§3.3/§3.5/§3.6/§3.7、§5、§6、§9）：训练路径不起 HTTP server，模型/采样器/老师都是 twinkle Ray actor（按 DeviceGroup 放置）；`vllm_mode='server'` 重释为 `disaggregated`；MOPD 多老师 = 多个冻结 Ray-actor 模型 `forward_only` 打分；`vllm_server_*`/`teacher_model_server`/`_Remote*Teacher` 移除。
2. **数据集格式契约**（§1.5 A–G）：SFT 纯文本/多模态、embedding、reranker、RL 五类行格式 + label mask / next-token shift / `_NO_SHIFT_TASK_TYPES`；两个不同的字段删除机制（dev 删原始源列 vs. twinkle 模型边界 `to_transformers_dict` 放行名单删未用编码键）。
3. **D6 格式主线与老师表示法**（§1.5 A0/A0.1、§3.2/§3.3/§3.4/§3.7、§2.F、§4 D6、§9 阶段一）：主推 OpenAI 标准 `messages`，内部流转 twinkle 原生 `Trajectory`/`InputFeature`，RL extras 走 `user_data`；机制完全复用 twinkle `sampler`+`Template`（token-in-token-out，`SampledSequence.new_input_feature`/`concat_input_feature`，不手搓 shift）；legacy 格式入口兼容但主推新格式；embedding/reranker = 多 `messages` 拼接 + 可选 `label`；**OPSD/MOPD 老师用「模型」`forward_only` 而非「sampler」**（模型 logits 更稳定，cookbook 的 sampler+prompt_logprobs 路已评估否决）。附 §1.5 A0.1 与 legacy 训练期格式的逐项兼容性 diff。
4. **类与复用设计 + RL 仅 Ray**（新增 §3.8、§1.4）：抽出 `TrainLoop` 主干 + `OnPolicyLoop`/`DistillLoop` 两级基类，把 5 个各自重复脚手架的 loop 收敛为"主干 + 命名 hook"；协作对象（`Rollout`/`Teacher`/`Reward`/`Advantage`）组合注入；每步数据收进单一 `RolloutBatch` 值对象（消灭平行 list 的 off-by-one 类 bug）；**RL 只在 Ray 下运行、不支持 torchrun**，RL loop 不带 rank 守卫分支。目标：好理解（主干一份）/好定制（改 hook 不碰别算法）/好调试（单值对象 + 协作对象可独立驱动）。

下一步：**阶段 0（纯读审计）已完成**——§2.0 矩阵 12 算法 × 7 维度全部收口为 ✓/✗/—，并产出按严重度排序的重构工单清单（W00–W32，映射到 §9 各阶段）。审计未改任何代码。

**审计四大结论**：
1. **两处崩溃级 bug**：W00/A1（`CheckpointEngineManager(colocate=)` 必 TypeError，GRPO/PPO/RFT 全踩）、W03（PPO 位置参数调 kw-only GAE，PPO 从未跑通）。
2. **一处静默训错**：W01（离线偏好 `chosen/rejected_labels` 从不 shift → dpo/kto/cpo/orpo/simpo 全在自见分布上训练）。
3. **GKD 老师抽象（W29）直接挡住 MOPD/OPSD 复用**：现为 `{str 哨兵, model actor, _RemoteGKDTeacher}` 联合类型 + 散落 `isinstance` 派发、多"老师"是按标签路由非加权；须先按 §3.8(2) 提炼统一 `Teacher` 协议。
4. **测试形同虚设**：两份 `test_recipes.py` 并存、`rl/test_recipes.py:711` 用已改名字段必 TypeError、PPO 张量测试因 twinkle 不可导入全 SKIP、GRPO/PPO/GKD 无 full-loop e2e、三新方法零覆盖 —— 上述崩溃/训错 bug 无一被现有测试捕获。

请确认**阶段一起点**。建议阶段一范围（详见 §9）：抽 `TrainLoop`/`OnPolicyLoop`/`DistillLoop` 基类 + `Teacher` 协议（W29）+ `RolloutBatch` 值对象 → 修 W00/A1、W01、W02、W16 → 新增 OPSD/MOPD/RFT（下沉 `MOPDLoss`）→ examples 冒烟。是否按此起点开始阶段一，或需调整优先级/范围？
