# 强化学习迁移功能基线（legacy `swift rl` / `swift rlhf` 完整功能清单）

> 目的：把 legacy 的 **RL（强化学习 / 人类对齐）** 这条训练链路的功能面**逐项列全**，作为迁移到 dev 分层后的**功能回归基线**。dev 侧目前只实现了一部分，本份用来对照「legacy 到底有哪些算法、哪些参数、哪些训练方式、支持哪些模型后端与多模态」，确保 dev 补齐时每一项都可回归、行为准确。
>
> 与 `TRAIN_MIGRATION.md`（pt/sft 训练链路）、`MODEL_MIGRATION.md`（模型）、`DATASET_MIGRATION.md`（数据集）、`ARGUMENTS_MIGRATION.md`（参数）、`PLUGIN_MIGRATION.md`（插件）配套：那几份记 pt/sft，本份记 **rl**。
>
> 命令说明：`swift rlhf` 已改名为 `swift rl`，`rlhf` 作为兼容别名保留（`swift/cli/main.py`、`swift/cli/_megatron/main.py` 同时注册 `rl` 与 `rlhf` 两个键，指向同一入口）。本文统一用 `swift rl`。
>
> 来源：`swift/cli/{rlhf.py,_megatron/rlhf.py,main.py}`、`swift/arguments/rlhf_args.py`、`swift/rlhf_trainers/*`、`swift/rl_core/*`、`swift/rewards/*`、`swift/rollout/*`、`swift/pipelines/train/{rlhf.py,kto.py}`、`swift/trainers/trainer_factory.py`、`swift/megatron/{arguments,pipelines,trainers}/*`、`swift/template/{base.py,template_inputs.py}`、`docs/source/Instruction/{GRPO,RLHF,Command-line-parameters}`、`docs/source/Customization/Custom-dataset.md`、`examples/train/{rlhf,grpo,multimodal}`。

---

## 0. 一句话结构

```
CLI (swift rl → swift/cli/rlhf.py；megatron rl → swift/cli/_megatron/rlhf.py)
  └─ Pipeline (SwiftRLHF，继承 SwiftSft；Megatron 侧 MegatronRLHF)
       ├─ 参数 (RLHFArguments = TeacherModelArguments + GRPOArguments + PPOArguments
       │                          + RewardModelArguments + SftArguments)
       ├─ 模型 (policy + 可选 ref / value / reward / teacher；transformers / unsloth / megatron)
       ├─ 模板 (args.get_template，按算法 set_mode：'rlhf' / 'kto' / 'train' / 'transformers')
       ├─ 数据 (load_dataset → encode；kto 额外走 prepare_kto_dataset 造 KL 批)
       ├─ tuner (TunerMixin.prepare_model → peft / unsloth / tuners_map；ref_adapters 支持)
       └─ Trainer (TrainerFactory 按 rlhf_type 选 trainer + config)
```

- **两条并行的独立栈**：
  - **Transformers / trl 栈**：`swift rl` → `swift.cli.rlhf` → `swift.pipelines.rlhf_main`（`SwiftRLHF`），trainer 在 `swift/rlhf_trainers/*`。
  - **Megatron 栈**：`megatron rl` → `swift.cli._megatron.rlhf` → `megatron_rlhf_main`（`MegatronRLHF`），trainer 在 `swift/megatron/trainers/*`。legacy 无 `--backend` 开关，靠**用哪个命令**区分（`setup.py:162-164`）。
- **unsloth 不是独立的 RL 后端**，而是一个模型加载 / LoRA 包装后端（`--tuner_backend unsloth`），底层仍跑 trl 的 DPO/KTO/CPO/ORPO/RM trainer（见 §7）。
- **v5 dev 栈**：`USE_SWIFT_V5=1` 时 `swift rl` 路由到 `swift.dev.cli.rlhf`；此时 `megatron rl` 会被拒绝并提示改用 `swift rl --backend megatron`（`swift/cli/main.py:46-62`）。这正是 dev 要把两条栈用单一 `--backend` 统一的原因。

### 0.1 算法分派（rlhf_type → trainer / config）

`swift/trainers/trainer_factory.py:13-45`，键取自 `args.rlhf_type`（无则退回 `task_type`）：

| rlhf_type | Trainer（transformers） | Config |
|---|---|---|
| `dpo` | `swift.rlhf_trainers.DPOTrainer` | `DPOConfig` |
| `orpo` | `ORPOTrainer` | `ORPOConfig` |
| `kto` | `KTOTrainer` | `KTOConfig` |
| `cpo` | `CPOTrainer` | `CPOConfig` |
| `rm` | `RewardTrainer` | `RewardConfig` |
| `ppo` | `PPOTrainer` | `PPOConfig` |
| `grpo` | `GRPOTrainer` | `GRPOConfig` |
| `gkd` | `GKDTrainer` | `GKDConfig` |

> `rlhf_type` 全集（`swift/arguments/rlhf_args.py:232`）：`Literal['dpo','orpo','simpo','kto','cpo','rm','ppo','grpo','gkd']`，默认 `dpo`。
> **`simpo` 无独立 trainer**：`_init_simpo`（`rlhf_args.py:482-490`）把 `rlhf_type` 改写成 `cpo`，`loss_type='simpo'`，`beta=2.0`（`simpo_gamma` 默认 1）。所以 SimPO 就是「CPO + `loss_type=simpo`」。

`get_training_args`（`trainer_factory.py:62-73`）用 `asdict(args)` 按目标 Config 的字段签名过滤——**没在 Config 上声明的超参会被静默丢弃**，迁移时要注意每个算法真正生效的超参子集（见 §9 支持矩阵）。

---

## 1. 算法逐项（离线偏好类 + PPO + GKD）

所有偏好类 trainer 都继承共享基类 `RLHFTrainerMixin`（`swift/rlhf_trainers/rlhf_mixin.py`），**PPO 除外**（只继承 `SwiftMixin + HFPPOTrainer`）。通用做法是 `del HFxxxTrainer.__init__`，让 MRO 落到 `SwiftMixin.__init__`（模板 / 数据驱动构造）。

### 1.1 DPO（`rlhf_type='dpo'`）—— 唯一 loss 完全自研的偏好算法

- 文件：`swift/rlhf_trainers/dpo_trainer.py`（471 行）；类 `DPOTrainer(RLHFTrainerMixin, SwiftMixin, DataLoaderMixin, HFDPOTrainer)`。
- 覆盖了 `concatenated_forward` / `dpo_loss` / `get_batch_loss_metrics` / `compute_ref_log_probs` / `compute_loss` 等整条 loss 路径。
- **`concatenated_forward`** 返回 dict（chosen/rejected logps、mean logits、nll_loss、MoE aux_loss）；含多模态 logits 尾切（`logits[:, -labels.shape[1]:]`）、`labels` 左移一位、IPO 长度归一、LD-DPO（`ld_alpha`，含 padding_free 的 `cu_seqlens` 路径与 dense 路径）。
- **`dpo_loss` 支持的 `loss_type` 全集**（`dpo_trainer.py:211-343`）：
  `sigmoid`（默认）、`robust`、`exo_pair`（`label_smoothing=0` 时强制 1e-3）、`hinge`、`ipo`、`bco_pair`（用 `RunningMoments`）、`sppo_hard`、`nca_pair`、`aot` / `aot_unpaired`、`apo_zero`、`apo_down`、`discopop`（受 `discopop_tau` 控制，默认 0.05）、`sft`（直接展开 nll_loss）；其余报错。
  另有 f-散度分支：`f_divergence_type ∈ {reverse_kl, alpha_divergence, js_divergence}` + `f_alpha_divergence_coef`。
- **MPO（混合偏好优化）**：`loss_type` 传多个值 + `loss_weights`（仅 DPO 支持，`_process_loss_type` 校验，`rlhf_args.py:315-337`；需 trl≥0.20）。
- **RPO**：`loss = dpo_loss + rpo_alpha * nll_loss`（`rpo_alpha` 非空时叠加 SFT 项）。
- **参考模型**：需要 ref；LoRA 训练可免 ref（`null_ref_context()` 用 `disable_adapter()` 拿基座权重，或 `ref_adapters` → `ref_adapter_name`）。`ref_adapters` 仅 dpo/kto/grpo 支持（`rlhf.py:189-190`）。
- 真正生效超参：`beta`、`label_smoothing`、`loss_type`(可列表)、`loss_weights`、`rpo_alpha`、`ld_alpha`、`discopop_tau`、`f_divergence_type`、`f_alpha_divergence_coef`、`precompute_ref_log_probs`、`ref_adapter_name`、`reference_free`、`router_aux_loss_coef`。
- 能力：padding_free/packing ✅、sequence_parallel ✅（**唯一支持 SP 的偏好算法**）、多模态 ✅。

### 1.2 CPO（`rlhf_type='cpo'`）

- 文件：`cpo_trainer.py`（39 行），`CPOTrainer(RLHFTrainerMixin, SwiftMixin, HFCPOTrainer)`（trl.experimental）。
- **断言无 ref_model**（reference-free）。swift 侧只设 `label_smoothing/loss_type/cpo_alpha/alpha`，loss 交给 trl 的 `cpo_loss`。
- trl `cpo_loss` 支持 `loss_type ∈ {sigmoid, hinge, ipo, simpo}`，且先做 AlphaPO 变换（`alpha≠0` 时 `r=(1-p^{-alpha})/alpha`）；`simpo` 分支按 `simpo_gamma/beta` 平移。`loss = losses.mean() + cpo_alpha * nll_loss`（`cpo_alpha` 默认 1.0）。
- 生效超参：`beta`、`loss_type`、`label_smoothing`、`cpo_alpha`、`simpo_gamma`、`alpha`、`max_completion_length`、`generate_during_eval`、`disable_dropout`。
- 能力：padding_free ❌、SP ❌、多模态 ✅（走 mixin 通用前向）。

### 1.3 SimPO（`rlhf_type='simpo'`）

- 见 §0.1：改写为 CPO + `loss_type='simpo'`、`beta=2.0`、`simpo_gamma=1`。用**长度平均**的 logp 差 + margin。`cpo_alpha>0` 即 CPO-SimPO 混合（加 NLL 正则），会告警。
- 无 ref / value / teacher。padding_free ❌、SP ❌、多模态 ✅。

### 1.4 ORPO（`rlhf_type='orpo'`）

- 文件：`orpo_trainer.py`（24 行），`ORPOTrainer(RLHFTrainerMixin, SwiftMixin, HFORPOTrainer)`。**断言无 ref_model**（odds-ratio loss 数学上 reference-free，用模型自身长度归一概率作隐式参考）。
- **关键 swift 钩子**（`rlhf_mixin.py:103-104`）：ORPO 时把 `concatenated_input_ids` 替换成 `concatenated_labels`，让 trl 的 NLL 用已 `-100` 掩码的 labels（尊重 `loss_scale` 的响应区掩码 / 多模态 / packing 布局），否则会连 prompt 一起训。迁移时**必须保留**。
- trl loss：`log_odds = (chosen_logps - rejected_logps) - (log1mexp(chosen) - log1mexp(rejected))`，`loss = nll_loss - beta*mean(logsigmoid(log_odds))`；`average_log_prob=True`（长度归一）。
- `ORPOConfig` **无 loss_type**。生效超参基本只有 `beta`。
- 文本 ORPO 的 `loss_scale` 默认为 `'default'`（多模态则 `'last_round'`，见 §8）；padding_free ❌、SP ❌、多模态 ✅。

### 1.5 KTO（`rlhf_type='kto'`）

- 文件：`kto_trainer.py`（133 行）+ 数据构造 `swift/pipelines/train/kto.py`（78 行）。`KTOTrainer(RLHFTrainerMixin, SwiftMixin, HFKTOTrainer)`。
- **数据是「prompt+response+label(布尔)」**，非成对偏好。`prepare_kto_dataset`（`kto.py:42-78`）用 `KTOPreprocessor` 把最后一条 assistant 轮转到队首造出**不成对的 KL 行**（`kl_messages=[messages[-1]]+messages[:-1]`），并校验 `desirable_weight/undesirable_weight` 比例。`loss_type='apo_zero_unpaired'` 时跳过 KL 位移。
- swift 只覆盖前向侧（`forward` / `_get_model_kwargs` / `get_batch_logps`(reduction='sum') / `_compute_kl_logps`），loss 交给 trl `kto_loss`：
  `chosen_losses=1-sigmoid(beta*(chosen_logratios-kl))`、`rejected_losses=1-sigmoid(beta*(kl-rejected_logratios))`，按 desirable/undesirable 加权；`apo_zero_unpaired` 分支不用 KL。
- 模板 mode `'kto'`（`_kto_encode` / `_kto_data_collator`）。ref 可用 LoRA 免加载。padding_free/packing ✅、SP ❌、多模态 ✅（模型侧；注意目前没有 VL 的 KTO 数据集，`examples/train/multimodal/rlhf/kto.sh` 是拿 VL 模型跑文本 KTO 数据）。
- liger：trl 支持，但 `_compute_loss_liger` 需 `ref_model` 存在且不支持 padding_free（迁移注意这两个隐性限制）。

### 1.6 RM（`rlhf_type='rm'`，奖励模型训练）

- 文件：`reward_trainer.py`（116 行），`RewardTrainer(RLHFTrainerMixin, SwiftMixin, HFRewardTrainer)`。
- `_init_rm`（`rlhf_args.py:492-495`）强制 `task_type='seq_cls'`、`num_labels=1`。
- **loss 自研 `compute_loss`**（Bradley-Terry）：`rewards=model(**inputs).logits`，chosen/rejected 各半，`loss=-logsigmoid(r_chosen - r_rejected - margin).mean()`；`center_rewards_coefficient` 非空时加 `coef*mean((r_chosen+r_rejected)^2)` 鼓励零均值。无 loss_type 变体。
- 数据：`messages + rejected_response`，可选每行 `margin`。模板 mode `'rlhf'`（seq_cls 分支会剥掉 labels）。无 ref/value/teacher。padding_free ❌、SP ❌、多模态 ✅。
- 生效超参几乎只有 `center_rewards_coefficient`、`disable_dropout`（trl `RewardConfig` 字段很少，其余被丢弃）。

### 1.7 PPO（`rlhf_type='ppo'`）—— 唯一的 value-model 算法、栈里的异类

- 文件：`ppo_trainer.py`（118 行），`PPOTrainer(SwiftMixin, HFPPOTrainer)`，**不继承 RLHFTrainerMixin**。
- 用「保存再删除」`ppo_trainer_init = HFPPOTrainer.__init__` 的手法：先跑 `SwiftMixin.__init__`（剥出 reward/value model），再显式调用 trl 的 init，并 `patch_getattr(..., 'policy')`。还 monkeypatch `DataLoader.__init__` 强制 swift 的 collate_fn。
- **需要四个模型**：policy、ref、reward、**value（可训练，唯一有 value model 的算法）**。`save_model` 只存 `unwrapped_model.policy`（不存 value head）。
- loss 完全交给 trl：`pg_loss + vf_coef*vf_loss`，GAE(`gamma`,`lam`)、KL 惩罚(`kl_coef`)、`cliprange`/`cliprange_value`、`whiten_rewards`、`missing_eos_penalty`；`logits /= temperature`。无 loss_type 变体。
- 模板 mode `'transformers'`，数据是**纯 prompt 的 messages**（响应在线生成），**强制左 padding**（`rlhf_args.py:465-468`，注释 `TODO: streaming, MLLM`——即多模态 PPO 实际未打通）。padding_free ❌、SP ❌、多模态 ⚠️ 实际仅文本。
- PPO 专属参数（`PPOArguments`，`rlhf_args.py:93-127`）：`num_ppo_epochs=4`、`whiten_rewards`、`kl_coef=0.05`、`cliprange=0.2`、`vf_coef=0.1`、`cliprange_value=0.2`、`gamma=1.0`、`lam=0.95`、`num_mini_batches=1`、`local_rollout_forward_batch_size=64`、`num_sample_generations=10`、`response_length`(废弃)、`missing_eos_penalty`。

### 1.8 GKD（`rlhf_type='gkd'`，广义知识蒸馏 / 在线蒸馏）

- 文件：`gkd_trainer.py`（533 行）+ `gkd_loss.py`（272）+ `gkd_helpers.py`（473）；`GKDTrainer(RolloutTrainerMixin, SwiftMixin, HFGKDTrainer)`（`RolloutTrainerMixin` 间接继承 `RLHFTrainerMixin`）。
- **需要 teacher**（本地权重 / vLLM server / HTTP API 三种来源），**无 ref、无 value**。teacher 三种模式（`TeacherModelArguments` 文档，`rlhf_args.py:40-91`）：
  - 不设 teacher：自蒸馏，teacher=当前 student；
  - teacher==model 且 LoRA：固定 teacher，用 `disable_adapter()` 取基座 logits（`_teacher_use_disable_adapter`，不额外加载）；
  - teacher≠model：独立冻结 teacher。
  - `teacher_model_server`：从远端 API 取 teacher logprobs（支持多 teacher，JSON `[{url,tags}]`，按 `teacher_tag_key` 路由）；此时**必须**设 `gkd_logits_topk`。
- **在线采样比例 `lmbda`**（默认 0.5）：`_get_random_num()`（seed=`args.seed+global_step`，可复现）≤ lmbda 走 student rollout，否则用数据集响应 teacher-forcing。
- **loss（`gkd_loss.py`）**：广义 JSD，由 `beta` 控制（`beta=0`→前向 KL，`beta=1`→反向 KL，中间为混合），按 `temperature` 缩放；`gkd_logits_topk` 非空时只取 teacher top-k logits（省显存 / 适配 API teacher）；`_align_vocab` 处理师生词表不一致；`sft_alpha>0` 时对 teacher-forcing 行叠加 NLL。
- liger：用 `LigerFusedLinearJSDLoss`，但**需显式 teacher、禁 `sft_alpha>0`、禁 `gkd_logits_topk`**（`_check_gkd`，`rlhf_args.py:748-772`）。
- 模板 mode `'train'`（非成对），**强制左 padding**；padding_free/packing ✅、SP ❌、多模态 ✅（`examples/train/multimodal/rlhf/gkd/*` 用 InternVL3 师生）。
- teacher 相关参数：`teacher_model`、`teacher_adapters`、`teacher_model_type/revision`、`teacher_deepspeed`、`teacher_model_server`、`offload_teacher_model`；GKD 专属：`beta`(默认 0.5)、`temperature`(0.9)、`lmbda`(0.5)、`sft_alpha`(0)、`seq_kd`(废弃，设 True 报错)、`gkd_logits_topk`(None)、`max_completion_length`(512)、`log_completions`、`offload_teacher_model`。

---

## 2. GRPO（`rlhf_type='grpo'`，在线 RL 主力）

文件：`swift/rlhf_trainers/grpo_trainer.py`（2605 行），`GRPOTrainer(RolloutTrainerMixin, SwiftMixin, HFGRPOTrainer)`。基于 trl 0.29 的 `HFGRPOTrainer`，但 loss/advantage/rollout 大量自研。

### 2.1 核心 loss 与 advantage

- **单步流水线**（`rollout_mixin.py:122-145`）：`_rollout_samples` → `_score_completions` → `_prepare_batch_inputs` → `_postprocess_batch` → `_log_rollout`；每 `steps_per_generation * num_iterations` 步重新生成。
- **per-token logps**：`_get_per_token_logps_and_entropies`（单次或分块前向，`logits[:, -(logits_to_keep+1):-1, :]`，`logits/=temperature`，`selective_log_softmax`）；SP 走 `_get_logps_via_sp`（`GatherLoss` + ring-parallel）；padding_free 走 `_unpad_logps_and_entropies`；liger 走 `LigerFusedLinearGRPOLoss`。
- **old / ref / teacher logps**（`_prepare_batch_inputs`，`no_grad`）：`old_per_token_logps` 仅当 `num_iterations>1` 或 `gas % steps_per_generation != 0` 时算；ref 在 `beta==0.0` 时为 None，否则用 ref_model 或 `null_ref_context()`；teacher 见 §2.2 OPD-RL。
- **KL**：loss 内用 **k3 估计**（`per_token_kl = clamp(exp(clamp(ref-logp,-20,20)) - (ref-logp) - 1, -10, 10)`，`beta≠0` 且非 `kl_in_reward` 时叠加）；`kl_in_reward=True` 时把 KL 直接从 reward 里扣（在 advantage 之前）。默认：grpo→False，rloo/reinforce++→True。`beta` 默认 None→0.04。
- **advantage 估计器**（`swift/rl_core/advantage.py::compute_advantages`，reward 按 `view(-1, num_generations)` 分组）：
  - `grpo`：`r - 组内均值`；
  - `rloo`：留一法基线 `K/(K-1)` 缩放；
  - `reinforce_plus_plus`：批级归一。
  - `scale_rewards ∈ {group, batch, none, gdpo}`：最终 `adv /= (std+1e-4)`；默认随估计器（grpo→group、rloo→none、reinforce++→batch）；`gdpo` 为**逐奖励函数**组内归一后加权求和再全局归一（需 reward_funcs，且不支持 `kl_in_reward=True`）。
  - `compute_advantages_dynamic`：动态采样下按 prompt_id/request_id 处理每组样本数不等的情形。
- **重要性采样层级** `importance_sampling_level`：`token`（默认，逐 token log-ratio）、`sequence`（GSPO，序列内掩码均值）、`sequence_token`（GSPO-token，detach 技巧：token 级梯度 + 序列级 IS 值）。
- **裁剪**：`epsilon`(0.2) 低裁、`epsilon_high`(None→epsilon) 高裁（clip-higher，DAPO）、`delta`（双侧裁剪，INTELLECT-2，`coef_1=clamp(coef_1,max=delta)`）；`loss=-min(coef_1*adv, coef_2*adv)`。

### 2.2 GRPO 的 `loss_type` 全集与算法变体

**`loss_type` 接受集**（`grpo_trainer.py:1096-1145`，其余报错）：
`grpo`（swift 默认，逐序列 token-mean→批 mean）、`bnpo`（token-sum/总 token 数）、`dr_grpo`（除以 `batch*max_completion_length` 常数，去长度/难度偏置）、`dapo`（全局 token-mean）、`cispo`（clamp IS 权重 detached × logp）、`sapo`（sigmoid 软门控）、`real`（组内对比 logsumexp）、`fipo`（grpo loss × 逐 token 影响权重）。
> 注意：trl 自身默认 `dapo`，**swift 覆盖为 `grpo`**。

**变体 → 开启方式 → 机制**（一行）：

| 变体 | 开启参数 | 机制 |
|---|---|---|
| DAPO | `loss_type=dapo` + `dynamic_sample` + `epsilon_high` + `overlong_filter`/`soft_overlong` | 全局 token-mean 归一 + clip-higher + 组内 std=0 重采样 + 超长过滤 |
| Dr.GRPO | `loss_type=dr_grpo`、`scale_rewards=none` | 常数分母，去长度/难度偏置 |
| GSPO | `importance_sampling_level=sequence` | 序列级 IS 比 |
| GSPO-token | `importance_sampling_level=sequence_token` | detach：token 梯度 + 序列 IS 值 |
| RLOO | `advantage_estimator=rloo` | 留一法基线 `K/(K-1)` |
| REINFORCE++ | `advantage_estimator=reinforce_plus_plus` | 批级归一，默认 `kl_in_reward=True` |
| SAPO | `loss_type=sapo`、`tau_pos=1.0`、`tau_neg=1.05` | 正/负优势分别用不同温度 sigmoid 软门控 |
| REAL | `loss_type=real`、`real_tau=0.5` | 组内对比 logsumexp；强制 `scale_rewards=none` |
| FIPO | `loss_type=fipo`、`fipo_decay_rate/clip_range/clip_high_only/safety_threshold` | 未来 KL 加权的逐 token 影响权重（detached） |
| CISPO | `loss_type=cispo` | clamp IS 权重 × logp，无比值裁剪 |
| CHORD | `chord_sft_dataset` + `chord_sft_per_device_train_batch_size` + `chord_mu_*` + `chord_enable_phi_function` | mu 调度混合 `loss=(1-mu)*grpo+mu*sft`，可选 phi=p(1-p) token 加权 |
| OPD-RL（在线策略蒸馏当 RL） | `teacher_model`/`teacher_model_server` + `teacher_kl_coef=1.0` | 逐 token reward 塑形 `adv + coef*(teacher_logp - old_logp)`；与 real/fipo/off-policy mask 不兼容 |
| RLSD（自蒸馏 RLVR） | `advantage_reweight=rlsd`、`rlsd_lambda=0.5`、`rlsd_reweight_clip_range`、`rlsd_lambda_warmup/decay_steps`、`rlsd_negative_only` | 用 teacher-vs-student logprob 差 `w=exp(sign(A)*Δ)` 逐 token 重加权 advantage；需 reward_funcs；与 real/fipo/liger 不兼容 |
| SDAR（自蒸馏智能体 RL） | `sdar_loss_coef>0`、`sdar_gate_beta=5.0` | 置信度门控蒸馏辅助损失 `coef*mean(sigmoid(gate_beta*Δ)*Δ)`；与 RLSD 互斥 |
| Rollout IS 修正 | `rollout_importance_sampling_mode ∈ {token_truncate,token_mask,sequence_truncate,sequence_mask}`、`rollout_importance_sampling_threshold=2.0` | 纠正训练/采样策略不一致（截断或掩掉超阈权重） |
| Off-policy 序列掩码 | `off_policy_sequence_mask_delta` | 序列均值 logp 偏移 > 阈值且 adv<0 时整序列置零 |
| 熵掩码（80/20 法则） | `top_entropy_quantile<1.0`、`log_entropy` | 只有高熵分位 token 进 loss |
| 双侧裁剪 | `delta` | `coef_1=clamp(coef_1,max=delta)`（INTELLECT-2） |
| 动态自蒸馏 OPSD | 数据含 `teacher_prompt` 列 | teacher 在特权 prompt 上前向（RLSD/SDAR/OPD-RL 共用） |
| 同步 ref | `sync_ref_model`、`ref_model_sync_steps=512`、`ref_model_mixup_alpha=0.6` | EMA：`ref=(1-alpha)*ref+alpha*policy`（ZeRO-3 感知） |
| off-policy 诊断 | `log_rollout_offpolicy_metrics=True` | 记录训练/采样 ppl、k1/k3 KL、ppl_ratio、chi2 |

> **动态采样（DAPO）**：`_dynamic_sampling` 保留组内 reward std>0 的组，从 `dynamic_resample_iterator` 补足，上限 `max_resample_times=3`，不足则告警回退。`resample.py::resample_encode_failed_inputs` 同时用于编码失败补样。
> **参数校验集中在** `rlhf_args.py`：`_init_grpo`(339-392)、`_check_grpo`(532-572)、`_check_opd_rl`(669-697)、`_check_rlsd`(574-627)、`_check_sdar`(629-667)。GRPO 强制：`num_generations>1`、`truncation_strategy ∈ {left,delete}`、不支持 `cached_dataset`、`gradient_accumulation_steps` 默认 1、trl≥0.20。

### 2.3 奖励函数（`swift/rewards/`）

**内置 ORM 注册表**（`orm.py:455-464`）：

| 名称 | 类 | 计算 |
|---|---|---|
| `accuracy` | `MathAccuracy` | math_verify 解析/校验，`<answer>` 标签或 boxed；无法解析标准答案→0 |
| `format` | `Format` | 正则校验 `<think>..</think><answer>..</answer>` → 1/0 |
| `react_format` | `ReActFormat` | 校验 ReAct 的 Action/Action Input 格式 |
| `cosine` | `CosineReward` | 长度-余弦奖励（正确/错误答案各一组端点值），用 response_token_ids |
| `repetition` | `RepetitionPenalty` | `(1 - 唯一 ngram/总 ngram) * max_penalty`（n_grams=3、max_penalty=-1.0） |
| `soft_overlong` | `SoftOverlong` | DAPO 软超长惩罚，需 `soft_cache_length`（`soft_max_length` 默认=max_completion_length） |
| `toolbench` | `ReactORM` | Action/Action Input 的 F1 + rouge-l 分级 |
| `math` | `MathORM` | boxed 抽取 + sympy 判等（或 OpenCompass 评估器） |

- 奖励函数签名 `__call__(self, completions, **kwargs)`；`kwargs` 来自数据集额外列（`to_reward_row` 会把 `messages/images/videos/audios/tools/objects` 及 `extra`（如 `solution`）全部展平传入）——**即奖励函数能拿到多模态输入**（见 §8.4）。异步奖励（`AsyncORM`）经 `asyncio.gather` 执行。
- **外部奖励模型**（`--reward_model`）：`rm_plugin.py` 的 `default`(`DefaultRMPlugin`，value head 取 `logits[:,0]`) 与 `genrm`(`GenRMPlugin`，LLM-as-judge，正则抓 `Reward: x`)。注册名取模型路径末段。作为**最后一个奖励源**参与 `reward_weights` 加权。
- **PRM**（`prm.py`：`qwen_max`/`client`）：文件头明确**不支持 GRPO 训练**（仅采样），迁移 GRPO 时排除。
- 组合：`rewards_per_func`（`[N, 奖励源数]`）× `reward_weights` 后 `nansum`；`reward_weights` 长度须等于「reward_funcs 数 + 外部 reward_model 数」。

### 2.4 Rollout / 生成

- **引擎模式**（`_prepare_vllm`）：
  - `colocate`：`GRPOVllmEngine`（external_launcher、TP 子组、`sleep_level` 显存释放、可选 bnb 量化 + LoRA）；
  - `server`：`VLLMClient`/`VLLMInferClient`（`swift rollout` 起的独立服务，HTTP `/infer/`，多 server 分片）；
  - 非 vLLM：`TransformersEngine` 兜底。
  - `use_vllm` 仅 `grpo`/`gkd` 支持（`rlhf_support_vllm_types=['grpo','gkd']`）；`vllm_mode` 与 `use_vllm` 必须成对；`async_generate` 要求 `vllm_mode=server`。
- **权重同步**（`_move_model_to_vllm`）：全量 vs adapter；server 全量走 `SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE`(默认 512MB) 分桶 + NCCL 广播；server LoRA 走 `update_adapter_(flattened_)param`；colocate LoRA 走 `add_lora(load_inplace)`；ZeRO-3 逐组 `GatheredParameters` gather（LLM 层分块、embed/lm_head 单批、vision tower 单独）；FSDP2 tensor 级 LoRA 合并。
- **异步生成** `async_generate`：`_prefetch` 提交到 executor → DataCache 队列，用上一步权重生成（近似，忽略 advantage 的 clip）；与 multi-turn 不兼容。
- **多轮** `multi_turn_scheduler` + `max_turns` + `completion_length_limit_scope`(`total`/`per_round`)：
  - colocate 驱动在 `swift/rollout/agent_loop.py::run_multi_turn`；server 模式在 `swift/rollout/multi_turn.py::MultiTurnScheduler`；
  - 内置 scheduler（`multi_turns` 注册表）：`math_tip_trick`、`thinking_tips_scheduler`、`gym_scheduler`、`openenv_scheduler`；
  - 逐轮累积 `response_token_ids`/`response_loss_mask`/`rollout_logprobs`，并**校验 logprob 数与 loss_mask==1 数一致**（不一致则清空 → 关闭 rollout IS）。
- **Gym 环境** `gym_env`/`use_gym_env`（`swift/rollout/gym_env.py`）：`Env` ABC（`async reset/step/close`），内置 `math_env`；奖励经 `rollout_infos['total_reward']` → `gym_reward` 列，无需 reward_funcs。
- **num_generations 分组**：`RepeatSampler`（每 prompt 连续重复 num_generations 次）；`prompt_id` 按 messages 的 JSON 归并（相同 prompt 共享 id），`request_id=chatcmpl-{uuid}`；advantage `view(-1, num_generations)`；评估用 `num_generations_eval`。
- **批大小关系**（`_init_generation_batch_params`）：`generation_batch_size` 与 `steps_per_generation` 二选一，须被 `num_processes*per_device_train_batch_size` 整除，且 `generation_batch_size % num_generations == 0`。
- vLLM 相关参数见 `VllmArguments`（`args_mixin.py:8-96`）：`vllm_gpu_memory_utilization`、`vllm_tensor_parallel_size`、`vllm_pipeline_parallel_size`、`vllm_enable_expert_parallel`、`vllm_max_num_seqs`、`vllm_max_model_len`、`vllm_limit_mm_per_prompt`、`vllm_max_lora_rank`、`vllm_enable_prefix_caching`、`vllm_quantization`、`vllm_reasoning_parser`、`vllm_speculative_config`、`vllm_engine_kwargs`、`vllm_data_parallel_size` 等。

### 2.5 GRPO 关键数据结构（`swift/rl_core/data.py`）

- `OnPolicySample`：`messages`/多模态/`tools`/`extra`/`prompt_id`/`request_id`/`response_token_ids`(逐轮)/`response_loss_mask`/`rollout_logprobs`/`finish_reason`/`rollout_infos`/`teacher_prompt`；方法 `build_teacher_view`(OPSD)、`to_reward_row`、`from_row`、`apply_rollout_output`、`to_infer_request`。
- `GRPOSample`(+rewards/advantages)、`GRPOBatch`（completion_mask、truncated_mask、old/ref/rollout/teacher logps、advantages、num_items_in_batch、logits_to_keep）；`GKDSample/GKDBatch` 同源。
- 协议：`RolloutInferRequest`（+images/data_dict/uuid）、`RolloutOutput`（response/messages/response_token_ids/response_loss_mask/rollout_logprobs/rollout_infos）。

---

## 3. 参考模型 / 奖励模型 / value / teacher 的装配

`swift/pipelines/train/rlhf.py`：
- `_prepare_single_model`（61-115）：ref/reward/teacher → `requires_grad_(False).eval()` + `use_cache=False`；**value model 保持可训练**。
- `_prepare_model_tokenizer`（117-173）：ref 在 gkd 时跳过；value 仅 ppo；teacher 仅 gkd/grpo；非 grpo 的 reward_model 收敛为单个。
- `prepare_model`（175-192）：`ref_adapters` → `adapter_name='ref_adapter'`，仅 dpo/kto/grpo。
- `_get_trainer_kwargs`（226-250）：按算法传 ref/reward/value/teacher、grpo/gkd 的 `vllm_client`、grpo 的 `reward_funcs`/`chord_sft_dataset`、teacher 相关（`teacher_deepspeed_config`/`teacher_model_server`/`teacher_use_disable_adapter`）、gkd 的 `gkd_logits_topk`。
- `RewardModelArguments`：`reward_model`(可多个)、`reward_adapters`、`reward_model_type/revision`、`reward_template`。

---

## 4. RLHFTrainerMixin 提供的共享行为

`swift/rlhf_trainers/rlhf_mixin.py`（195 行，DPO/CPO/ORPO/KTO/RM 直接用，GKD 经 RolloutTrainerMixin 间接用，**PPO 不用**）：
- `concatenated_forward`（84-119）：**单次真实前向 + monkeypatch** 技巧——先在拼接批上跑一次真前向，再临时把 `self.concatenated_inputs`/`model.__call__` 打桩返回缓存输出，让 trl 原生的 `concatenated_forward` 复用而不重复前向。依赖 `_rlhf_data_collator` 的「chosen 行在前、rejected 行在后」约定（`batch_size = attention_mask.shape[0]//2`）。含 MoE router logits、多模态 logits 尾切、ORPO 的 labels 透传。
- `get_per_token_logps`（136-183）：SP==1 走 trl `selective_log_softmax`；SP>1 走 `GatherLoss` + `sequence_parallel.gather`。
- `null_ref_context`（185-194）：LoRA 免 ref——PEFT `disable_adapter()` 或切到 `ref_adapter_name`。
- `create_loss_and_eval_metric → {}`（**故意不接 swift 的 loss_map/eval_metrics_map**）；`_prepare_inputs` 接 SP；`get_train_dataloader` 注入 rank-aware 的 worker_init_fn；`compute_loss` 按 grad-accum 归一。
- **不消费 `loss_scale` 张量**（二值时置 None，掩码在 encode 期用 `labels==-100` 完成，见 §5）。

---

## 5. 数据格式与模板 mode

模板 mode 选择（`rlhf.py:194-201`）：`{'kto':'kto','gkd':'train','ppo':'transformers','grpo':'train'}`，默认 `'rlhf'`（dpo/cpo/orpo/simpo/rm）。Megatron 侧（`megatron/pipelines/train/rlhf.py:36-39`）：`{'grpo':'train','gkd':'train','kto':'kto'}`，默认 `'rlhf'`。

encode 侧（`swift/template/base.py`）：`_rlhf_encode`(484-502)/`_kto_encode`(504-507)；`loss_scale<=0` 的 token `labels=-100`（响应区由此界定），二值 loss_scale 时张量置 None。`_rlhf_data_collator`(1731-1747) chosen 前 rejected 后 + float `margin`；`_kto_data_collator`(1749-1771) 造 `completion_*`/`KL_completion_*` + 布尔 `label` 列表。

| rlhf_type | 模板 mode | 数据列 | 备注 |
|---|---|---|---|
| dpo / cpo / orpo / simpo | rlhf | `messages`（末条 assistant=chosen）+ `rejected_response`(str 或消息列表) 或完整 `rejected_messages` | `_compat_rejected_response` 在最后 user 轮拼接 rejected；断言 chosen≠rejected |
| rm | rlhf(seq_cls) | 同 dpo + 可选 `margin` | seq_cls 分支剥掉 labels |
| kto | kto | `messages`(prompt+response) + `label`(布尔) | `prepare_kto_dataset` 造 KL 批；`apo_zero_unpaired` 跳过 |
| ppo | transformers | 纯 prompt 的 `messages`（响应在线生成） | 强制左 padding；实际仅文本 |
| grpo | train | 纯 prompt 的 `messages` + **透传列**（如 `solution`）或 reward_model 列 | 保留全部额外列并转发给 ORM；用 `--reward_model` 时无需 chosen/rejected |
| gkd | train | `messages`（响应可选） | `lmbda>0` 时丢数据集响应改由 student 生成；teacher 由 `--teacher_model`/`--teacher_model_server` 提供 |

- 多模态偏好对：加 `images`（可选 `rejected_images`/`rejected_messages`）；仅给 `rejected_response` 时，rejected 侧媒体默认复用 chosen 侧；给完整 `rejected_messages` 则须自备 `rejected_*` 媒体。
- `teacher_prompt` 是标准列，供 GKD/OPD-RL 及 GRPO 的 RLSD/SDAR 使用。
- 注册数据集与列重映射见 `swift/dataset/data/dataset_info.json`、`swift/dataset/dataset/{llm,mllm}.py`（如 RLAIF-V 把 `chosen→response`、`rejected→rejected_response`）。

---

## 6. 能力门控（padding_free / packing / sequence_parallel / liger）

集中在 `swift/arguments/rlhf_args.py`：
- `_check_padding_free`（709-716）：padding_free/packing 仅 `['grpo','dpo','kto','gkd']`。
- `_check_sequence_parallel`（718-724）：`sequence_parallel_size>1` 仅 `['grpo','dpo']`。
- `_init_padding_side`（465-468）：ppo/gkd 强制左 padding。
- liger：DPO 无专属 liger 路径（靠 SwiftMixin 模型 patch）；KTO 走 trl liger（需 ref、不支持 padding_free）；GKD 用 `LigerFusedLinearJSDLoss`（需显式 teacher、禁 sft_alpha/topk）；GRPO 用 `LigerFusedLinearGRPOLoss`（要求 `advantage_estimator=grpo`、不支持 padding_free/双侧 delta/SP/熵掩码/off-policy mask）。

---

## 7. 后端 / 模型家族支持矩阵

### 7.1 rlhf_type × 后端

| rlhf_type | Transformers/trl | Megatron | 备注 |
|---|---|---|---|
| dpo | ✅ | ✅ | |
| kto | ✅ | ✅ | |
| grpo | ✅ | ✅ | |
| gkd | ✅ | ✅ | |
| rm | ✅ | ✅ | 置 seq_cls/num_labels=1 |
| cpo | ✅ | ❌ | |
| orpo | ✅ | ❌ | |
| simpo | ✅（改写→cpo） | ❌ | |
| ppo | ✅ | ❌ | 仅 trl（HFPPOTrainer） |

- Megatron 分派：`swift/megatron/pipelines/train/rlhf.py:17-34` 内联 `trainer_mapping={'dpo':'MegatronDPOTrainer','gkd':'MegatronGKDTrainer','grpo':'MegatronGRPOTrainer','kto':'MegatronKTOTrainer','rm':'MegatronRewardTrainer'}`，其余报 `ValueError`；Megatron 的 `rlhf_type` 收窄为 `Literal['dpo','kto','grpo','gkd','rm']`（`megatron/arguments/rlhf_args.py:10`）。
- Megatron 专属参数：`MegatronRLHFArguments`（`loss_scale='last_round'`、`truncation_strategy` 默认 grpo→left 其余→delete、`calculate_per_token_loss`）；ref 相关在 `megatron_args.py`（`mcore_ref_model`/`mcore_ref_adapter`/`ref_adapters`/`use_cpu_initialization`）。`MegatronRLHFTrainer` 仅在 `tuner_type='full'` 且 `rlhf_type∉{rm,gkd}` 时建 `ref_models`；grpo/gkd 用 `identity_data_collator`（rollout 阶段再拼）；per-token logps 含 context-parallel all-reduce。Megatron 的 vLLM client 仅 server 模式（grpo/gkd）。

### 7.2 unsloth

- **unsloth 是 tuner/模型后端，不是 RL 算法后端**：`--tuner_backend unsloth`（`base_args.py:92`）。`get_model_processor` 命中时走 `load_by_unsloth`（`swift/model/register.py:45-97`，按多模态/MoE/语言选 `FastVisionModel`/`FastModel`/`FastLanguageModel`，支持 4/8-bit 与全参）；LoRA 包装在 `swift/pipelines/train/tuner.py:205-219`。
- **无 unsloth 专属 RLHF trainer**：`TrainerFactory` 恒映射到 trl 系 trainer。所以「unsloth RLHF」= 用 unsloth 加载/包模型，再跑标准 trl 的 DPO/KTO/CPO/ORPO/RM。grpo/gkd/ppo 实际不走 unsloth（grpo/gkd 用自研 rollout 引擎，ppo 用 HFPPOTrainer）。
- 仅两处 unsloth 运行时钩子在共享 trainer：`seq2seq_trainer.py:110,222`（跳过 `logits_to_keep` 与 loss 重算）。
- 现状：**没有 unsloth 的 RLHF 示例脚本**，唯一 unsloth 示例是 SFT（`examples/train/tuners/unsloth/train.sh`）。

### 7.3 后端选择逻辑

- transformers vs megatron = **用哪个命令**（`swift rl` / `megatron rl`），legacy 无 `--backend` 开关。
- unsloth 与上正交，由 `--tuner_backend unsloth` 决定。
- `USE_SWIFT_V5` 把 `swift rl` 改路由到 `swift.dev.cli.rlhf`，并拒绝 `megatron rl`（提示用 `swift rl --backend megatron`）——即 dev 用单一 `--backend` 统一两条栈。
- `NPROC_PER_NODE`/`NNODES` 触发 `torch.distributed.run` 包装；`megatron rl` 另支持 `--use_ray`。

---

## 8. 多模态支持

### 8.1 核心机制

多模态**几乎全部由模板层承担**，不是各算法分别写分支：encode 经 `Template._encode_truncated → _rlhf_encode/_kto_encode` 产出含 `pixel_values`/`image_grid_thw` 等的 `chosen_*`/`rejected_*` 张量；`pre_forward_hook → _post_encode` 把 input_ids 转 `inputs_embeds`。偏好类 trainer（dpo/cpo/orpo/simpo/kto/rm）与 gkd/ppo **没有 is_multimodal 分支**，透明继承模板的多模态编码。

### 8.2 只有 GRPO 有显式多模态代码

因为它做 vLLM rollout + 分块前向 + 动态采样：`self.is_multimodal=model.model_meta.is_multimodal`（`grpo_trainer.py:106`）、动态采样重编码（769/775/2045）、`_get_last_hidden_state` 的多模态前向（1733-1741）、分块时跳过 `shape[0]!=batch_size` 的多模态张量（2020-2021，pixel 张量不可逐样本切）。Megatron GRPO 无 is_multimodal（模板 + rollout request 处理），ViT 梯度检查点在 `megatron/trainers/base.py:593-595`。

### 8.3 算法 × 多模态支持矩阵

| 算法 | 多模态 | 证据 / 示例 |
|---|---|---|
| dpo | ✅ | `examples/train/multimodal/rlhf/dpo/{lora,full}.sh`（Qwen2.5-VL）；Megatron `examples/megatron/multimodal/dense/dpo.sh`、`moe/full_dpo_offload.sh` |
| cpo/orpo/simpo | ✅ | 上述 dpo/lora.sh 注释「cpo/orpo/simpo/rm 也支持」；ORPO 多模态 loss_scale 分支 |
| rm | ✅ | 同 dpo/lora.sh 注释（seq_cls rlhf encode） |
| kto | ✅（模型侧） | `examples/train/multimodal/rlhf/kto.sh`（Qwen2.5-VL），但无 VL 的 KTO 数据集，是拿 VL 模型跑文本 KTO 数据 |
| gkd | ✅ | `examples/train/multimodal/rlhf/gkd/{fast,full}.sh`（InternVL3-2B 学生 + 8B 教师） |
| grpo | ✅（图像 + omni/音频） | `examples/train/grpo/internal/vllm_vl7b.sh`、`vllm_lora_qwenvl72b.sh`、`examples/train/grpo/qwen2_5_omni/grpo.sh`（`ENABLE_AUDIO_OUTPUT=1`）；Megatron `examples/megatron/grpo/{dense_colocate,sapo}.sh` |
| ppo | ⚠️ 实际仅文本 | 模板 mode `'transformers'`，`_init_padding_side` 强制左 padding 且注释 `TODO: streaming, MLLM`；无 VL PPO 示例 |

### 8.4 奖励函数与多模态

- 奖励函数经 `to_reward_row()`（`data.py:128-148`）把 `messages/images/videos/audios/tools/objects` + `extra` 全部作为 kwargs 传入，**即奖励函数能收到图像/视频/音频**。内置 ORM 主要消费 `solution`/`completions`，不直接用图像；奖励模型插件路径（`rm_plugin.py`）会把 `images` 等列转发。
- 自定义插件可用图像：`examples/train/grpo/plugin/deepeyes/deepeyes_plugin.py` 的 `DeepEyesReward` 读 `infer_request.images[0]`、裁剪后把新图追加回 rollout——这是多模态工具调用 RL 的参考实现。

### 8.5 vLLM rollout 传多模态

- `VllmArguments` 的 `vllm_limit_mm_per_prompt`（JSON）、`vllm_mm_processor_cache_gb` 经 `get_vllm_engine_kwargs` 传给引擎。
- 单请求多模态：`OnPolicySample.to_infer_request`（`data.py:244-294`）把 images/videos/audios/tools/objects 映射进 `RolloutInferRequest`，`{bytes/path}` 图像归一为 base64/path。
- rollout 后重置多模态缓存：server 走 `vllm_client.reset_mm_cache`，colocate 走 `engine.engine.reset_mm_cache`。
- 音频：`use_vllm` 时设 `SWIFT_AUDIO_LOAD_BACKEND='soundfile_pyav'` 对齐 vLLM 音频加载。
- **ORPO 多模态 loss_scale 分支**（`_set_loss_scale`，`rlhf_args.py:300-313`）：`orpo` 且非多模态→`'default'`；多模态 ORPO 落 `'last_round'`（避免多模态前向 padding labels，有些模型不扩展 image-pad token）。
- **GRPO 多模态截断注意**（`rlhf_args.py:155-161`）：多模态左截断可能剪掉多模态 token 致形状不匹配，建议 `truncation_strategy='delete'`（会重采样补样）。

---

## 9. 总支持矩阵（迁移速查）

| 特性 | dpo | cpo | simpo | orpo | kto | rm | ppo | gkd | grpo |
|---|---|---|---|---|---|---|---|---|---|
| RLHFTrainerMixin | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅(间接) | ✅(间接) |
| loss 自研 | ✅(全) | trl | trl | trl | trl | ✅ | trl | ✅ | ✅ |
| ref_model | ✅/LoRA免 | ❌ | ❌ | ❌ | ✅/LoRA免 | ❌ | ✅ | ❌ | ✅/LoRA免(beta=0则无) |
| value model | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| teacher | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | (reward) | ✅ | ✅(OPD-RL) |
| 模板 mode | rlhf | rlhf | rlhf | rlhf | kto | rlhf | transformers | train | train |
| padding | right | right | right | right | right | right | left | left | - |
| padding_free/packing | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| sequence_parallel | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| liger | - | - | - | - | ✅(需ref/无pf) | - | - | ✅(需teacher) | ✅(限制多) |
| MoE aux loss | ✅ | ✅ | ✅ | ✅ | ✅ | - | - | - | - |
| 多模态 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️文本 | ✅ | ✅ |
| Megatron 后端 | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| loss_type 变体数 | 15+ | 4 | 固定 | 无 | 2 | 无 | 无 | JSD(beta) | 8 |

---

## 10. 迁移到 dev 时要特别当心的实现要点

1. **单前向 monkeypatch**（`RLHFTrainerMixin.concatenated_forward`）：复用 trl loss 但只跑一次拼接前向，依赖「chosen 前 / rejected 后」的批约定与 `batch_size//2`。dev 若不复用 trl，要自己实现等价的拼接前向 + 掩码。
2. **completion_mask 的两套帧**：HF 未 roll 帧（`labels[:, -ltk:] != -100`）vs Megatron/Ray roll 帧（`roll(labels,-1) != -100`）——差一位就是 off-by-one（与既有 task #18 相关）。
3. **loss_scale 不消费张量**：偏好类的响应区掩码在 encode 期就写进 `labels==-100`；dev 不能指望运行时读 loss_scale 张量。
4. **ORPO labels 透传**、**SimPO=CPO(loss_type=simpo)**、**RM 强制 seq_cls/num_labels=1**、**PPO 只存 policy 不存 value head**、**GKD lmbda 随机数按 seed+global_step 可复现**——都是易漏的行为细节。
5. **GRPO 默认与 trl 不同**：swift 把 `loss_type` 默认设为 `grpo`（trl 是 `dapo`）；`kl_in_reward`/`scale_rewards` 默认随 `advantage_estimator` 变；`loss_type=real` 强制 `scale_rewards=none`；`gdpo`+`kl_in_reward` 会告警并关 kl_in_reward。
6. **rollout logprob 校验**：数量对不上会**静默清空**并关闭 rollout IS，不报错——迁移时建议改为可观测。
7. **参数静默丢弃**：`get_training_args` 按 Config 字段过滤，未声明超参被丢；dev 用 `parse_configs_strict` 严格校验，需保证每个算法的 Config 覆盖其真正生效的超参集。
8. **不支持组合**：unsloth × {grpo,gkd,ppo}、Megatron × {cpo,orpo,simpo,ppo}、PPO × 多模态（实际）、prm × GRPO 训练、`async_generate` × multi-turn——迁移文档/校验里应显式拒绝而非静默降级。
9. **teacher 三来源 + 多 teacher 路由**（本地/disable_adapter/vLLM server/HTTP API，按 `teacher_tag_key` 路由）是 GKD/OPD-RL/RLSD/SDAR 共用的一套装配，dev 需要统一抽象。
10. **奖励函数拿得到多模态**：`to_reward_row` 展平全部列，dev 的 reward 接口须保留这一透传契约。

---

## 11. 关键文件索引

- 参数：`swift/arguments/rlhf_args.py`（`RLHFArguments`/`GRPOArguments`/`PPOArguments`/`RewardModelArguments`/`TeacherModelArguments`）、`swift/rlhf_trainers/args_mixin.py`（`VllmArguments`/`RolloutTrainerArgumentsMixin`/`GRPOArgumentsMixin`）、`swift/rlhf_trainers/arguments.py`（各 `*Config`）。
- 流水线：`swift/pipelines/train/rlhf.py`（`SwiftRLHF`）、`swift/pipelines/train/kto.py`、`swift/pipelines/utils.py`。
- Trainer：`swift/rlhf_trainers/{dpo,cpo,orpo,kto,reward,ppo,grpo,gkd}_trainer.py`、`rlhf_mixin.py`、`rollout_mixin.py`、`base_rollout_mixin.py`、`vllm_client.py`、`gkd_loss.py`、`gkd_helpers.py`、`utils.py`。
- RL 核心：`swift/rl_core/{data,grpo_algorithm,advantage,resample}.py`；奖励：`swift/rewards/{orm,prm,rm_plugin}.py`；rollout：`swift/rollout/{agent_loop,multi_turn,gym_env,openenv_wrapper}.py`。
- 分派：`swift/trainers/trainer_factory.py`；CLI：`swift/cli/{main.py,rlhf.py}`、`swift/cli/_megatron/{main.py,rlhf.py}`、`swift/dev/cli/rlhf.py`。
- Megatron：`swift/megatron/arguments/rlhf_args.py`、`swift/megatron/pipelines/train/rlhf.py`、`swift/megatron/trainers/{rlhf_mixin,grpo_trainer,gkd_trainer,dpo_trainer,kto_trainer,reward_trainer,rollout_mixin,base}.py`。
- unsloth：`swift/model/register.py:45`（`load_by_unsloth`）、`swift/pipelines/train/tuner.py:205`。
- 模板：`swift/template/base.py`、`swift/template/template_inputs.py`。
- 文档：`docs/source/Customization/Custom-dataset.md`（RLHF 段 113-191）、`docs/source/Instruction/RLHF.md`、`docs/source/Instruction/GRPO/*`、`docs/source/Instruction/Command-line-parameters.md`。
- 示例：`examples/train/rlhf/*`、`examples/train/grpo/*`、`examples/train/multimodal/rlhf/*`、`examples/megatron/{grpo,multimodal}/*`。
