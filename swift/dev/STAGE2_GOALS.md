# 第二轮目标（Stage 2）

> 与 `ARGUMENTS_MIGRATION.md` / `MODEL_MIGRATION.md` / `DATASET_MIGRATION.md` / `PLUGIN_MIGRATION.md` / `PATCH_INVENTORY.md` / `DATASET_REDESIGN.md` 同级。
>
> 第一轮聚焦「把 legacy 的字段/数据集/模型/插件迁进 `swift/dev`，并把建模与训练引擎委托给 twinkle + mcore-bridge」，产出的是逐域迁移台账。
> **第二轮的中心从「迁移覆盖」转向「组件质量与可复用」**：不再只问「搬过来了没有」，而是问「这个组件能不能独立跑对、能不能被 twinkle / twinkle-cs 直接拿去用、有没有和 twinkle 的底层重复造轮子」。
>
> 本文件只登记目标与其约束边界，具体的缺口清单、落点与验收在各域文档或后续小节中展开。

---

## 目标一：组件可完整回归 swift 原有能力

`swift/dev` 的组件要能把 swift 原有的能力**完整**跑回来，而不是只覆盖到「能跑通」的子集。

**约束边界（哪些不计入「完整」）：**
- **已约定丢弃的不算**：第一轮各域文档里已判定「不迁 / 放弃支持 / 无消费方且属新设计」的项（例如 tuner 系列、lmdeploy、UI、被版本淘汰的模型、放弃的 SP 变体等），不纳入回归范围。
- **写得啰嗦的代码不算**：以「行为等价」为准，不以「逐行照搬 legacy 实现」为准。legacy 里冗余、绕路、历史包袱式的写法，允许在 dev 侧用更简洁的等价实现替代——只要外部可观测行为一致即视为已回归。

**衡量方式：** 逐域对照 legacy 的实际能力面，扣除上述两类后，差集应为空；差集非空的每一项都要在对应域文档里带原因登记，不允许悄悄缺失。

## 目标二：组件可用于 twinkle 流程，进而可用于 twinkle-cs 流程

`swift/dev` 的组件（尤其是模型、数据集）要能作为标准训练组件**接入 twinkle 的流程**；由于 twinkle-cs（client-server）构建在 twinkle 之上，组件一旦能被 twinkle 流程消费，就能被 twinkle-cs 流程消费。

**目标形态：** 后续 twinkle-cs 可以**直接选择 swift 的模型、数据集**作为训练组件，无需为此在 swift 侧另开一套适配。

**约束边界：** 组件对外暴露的接口/契约要贴合 twinkle 的既有约定（loss / optimizer / metric / advantage / dataloader 等内核的调用形状），而不是让 twinkle 去迁就 swift 的私有形状。

## 目标三：twinkle 底层组件尽量复用在 swift 中

凡 twinkle 已经提供的底层构件（loss、optimizer、metric、advantage、kernel、dataloader、权重同步等），swift 侧**优先复用**，不重复实现。

**判定原则：**
- swift 只保留「swift 特有的业务语义」（名称解析、格式归一、模板、recipe 编排、配置体系等），底层计算/训练内核交给 twinkle。
- 出现「swift 与 twinkle 各有一份等价实现」时，以复用 twinkle 为默认取向；确需在 swift 侧另立的，要写清 twinkle 现有组件为何表达不了（签名/语义/加载机制等具体原因），比照 `PLUGIN_MIGRATION.md` 中「为什么不复用 twinkle 的 loader」的登记方式。

## 目标四：组件单独可运行、无 bug、跨模态稳定

每个组件要能**独立运行**并通过验证，而不是只在完整训练链路里才被间接触达。

**三条具体要求：**
- **单独可运行**：组件有可独立触发的入口/测试，不依赖跑完整个 recipe 才能验证其正确性。
- **无 bug**：组件级测试通过，行为符合预期。
- **跨模态稳定**：纯文本与多模态两条路径都要稳定——多模态（图/视频/音频的加载、预处理、模板编码、打包等）不能只在纯文本下验证通过就算数。

---

## 与第一轮的关系

- 第一轮的迁移台账（各域 `*_MIGRATION.md`）是**覆盖面**的账本；第二轮在其之上补**质量与复用**的账本。
- 第一轮标注为「已迁字段/已注册但尚无消费方 / 仅推导未接线 / 底层在 twinkle 而 dev 未接线」的项（如 VPP 仅推导未转发建模、多轮 RL 与 gym 在 twinkle_agentic 有底层而 dev 无消费方等），正是第二轮目标一与目标二要正面处理的对象。
- 凡涉及 GPU / 多卡 / 跨模态的正确性，均以真实环境验证为准，不以「代码路径存在」为已完成。

---

## 进度与讨论记录（2026-09）

### 已落地：`swift sample` 的多轮 rollout（对应目标一 / 目标三）

- `swift sample` 现已复用 `swift/dev/rollout/multi_turn.py::MultiTurnRollout`（twinkle 多轮引擎适配层）与**现有 sampler 实例及其 device_mesh**，不二次加载权重；`run_grpo` 早已走同一套机制，sample 与之对齐。
- `num_return_sequences` 通过复制 trajectory 生成多条完整轨迹，底层每轮仍 `num_samples=1`。
- 评分契约按 `multi_turn_enabled` 分叉：多轮下 reward callable 收完整 `messages` + `rollout_infos` + `truncated`，单轮沿用旧 `completions` 入参。
- gym 环境启用时候选直接按 `rollout_infos['total_reward']` 排序；`score_ground_truth` 与 `use_gym_env` 互斥（合成的参考答案没有 rollout 轨迹）。
- 输出：多轮走完整 `messages` + `rejected_messages`（两条轨迹可能中途分叉），单轮保留旧 `rejected_response` 字符串。
- 边界：client 后端（外部 OpenAI 兼容 API）缺 token/logprobs/template 协议，validate 层显式拒绝其 scheduler 多轮模式。
- 状态：改动位于 `cli/sample.py`、`config/process.py`、`config/validate.py`、`recipe/run_sampling.py`，4 文件过 AST；未跑测试/lint（按约束待授权）。这兑现了本文「多轮 RL 与 gym 在 twinkle_agentic 有底层而 dev 无消费方」的 stage-2 目标——sample 侧现已成为消费方。

### 待处理：`_ModelReward` 越层依赖 legacy（对应目标三）

- 现状：`recipe/run_sampling.py::_ModelReward` 由 `SamplingConfig.orm_model / prm_model` 触发，直接 import legacy `swift.infer_engine.TransformersEngine`，绕过 dev builders 与 twinkle；Ray 下在 driver 创建、不属任何 DeviceGroup，可能与 sampler 抢卡；`prm_model` 被硬编码 `task_type='seq_cls'`（并非真 PRM 推理）；文件注释「评分路径无本地模型」与实现直接矛盾。
- 硬约束（已核实）：`build_sampler` 只支持 `task_type='causal_lm'`，池化前向（seq_cls/reranker/embedding）没有 sampler，打分必须走 `build_model(task='seq_cls')`。故「复用 sampler 对象本身打分」不可行；能讨论的只是复用**权重 / DeviceGroup**。
- 分场景方案（按 RM 相对 sampler 底模的形态）：
  - **A 独立本地模型**：`build_model(task='seq_cls')` 放独立 DeviceGroup（orm/prm），不复用。
  - **B RM = sampler 底模本身**：需 twinkle 支持在同一份权重上挂 seq_cls 头，否则只能按同路径二次加载（费显存，非真复用）。
  - **C RM = 底模的 LoRA**：底座共享、adapter 不同；sampler 已支持 `adapters=` / `enable_lora`，但打分仍需前向路径。
  - **D 远程 API**：无本地前向、无 GPU，做成独立 API reward plugin（类比 `_ClientSampler`）。
- 配置面缺口：`orm_model / prm_model` 目前是裸字符串路径，**无法表达 A/B/C/D**。需先决定表达方式（裸路径→A、URL 前缀→D、哨兵值或新增 `reward_source` 字段→B/C）。
- 两个待拍板点：(1) B/C 的真权重复用需先核实 twinkle 是否支持同权重 seq_cls 前向 / 权重共享，否则 B/C 退化为 A；(2) 配置面用前缀 / 哨兵判别，还是新增显式 `reward_source` 字段。
- 建议落地顺序：先做 A（独立本地模型 `build_model` + 独立 DeviceGroup）+ D（远程 API plugin），覆盖绝大多数真实用法；B/C 待 twinkle 能力核实后再定真做或退化。

### 下一个议题（未展开）

- vLLM 与 SGLang 后端对 Embedding / reranker / seq_cls 任务的支持情况——待核实两个 engine 各自的 pooling 任务能力面，再决定 dev 侧 `build_sampler` 的 `task_type` 限制如何放开或如何路由到 `build_model`。
