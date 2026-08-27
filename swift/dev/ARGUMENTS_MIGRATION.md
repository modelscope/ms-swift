# 参数迁移结果表（legacy `swift/arguments/` → `swift/dev/config/`）

> 记录 legacy 参数字段迁移到 dev config 的结论与依据。与 `MODEL_MIGRATION.md`、`DATASET_MIGRATION.md` 并列：模型看那份、数据集看那份、参数看这份。
>
> `swift/dev/config/` 按**类型**聚类（15 个 Config dataclass），因此迁移也按类型进行 —— 每个 legacy 字段落到与其语义对应的 Config，而不是按 legacy 文件搬。

# 参数迁移结果表

> `swift/arguments/` 与 `swift/megatron/arguments/` 的字段迁移均已完成。实测：dev 从 14 个 config / 410 字段变为 **18 个 config / 643 字段**；legacy-only 从 114 降到 56，megatron 独有缺口从 128 降到 **0**。
> 剩下的 56 项全部是按规则或改名判定不迁的，逐类列于第一节。**后处理（推导/校验）已迁一批** —— 见第五节；剩余的纯副作用（环境变量/分布式初始化/下载/插件导入）仍按规则 11 不迁。
>
> **`swift/megatron/arguments/` 的 128 项已全部迁入**（首批 18 + 次批 110），新建 `MoEConfig` 与 `MegatronConfig`。见第零节。

## 全参数面完整性审计（机械校验）

逐字段检查“要么已在 dev、要么在本文档有记录”，不满足者计为遗失：

| 参数来源 | 未进 dev 的数量 | 文档未记录 | 结论 |
|---|---|---|---|
| `swift/arguments/`（327 字段） | 56 | **0** | 全部有据 |
| HF `Seq2SeqTrainingArguments`（117，dev 已覆盖 92） | 25 | **0** | 全部有据 |
| `swift/megatron/arguments/`（335 字段，独有 247） | **0** | **0** | 全部迁入 |

上一版本本行为「128 / 未记录 128」—— 那是本文档真正的遗失，现已迁完。

校验方式：对每个未进 dev 的字段名，在本文档里正则搜索 `` `name` `` 是否存在。

## 本轮实际落地（与下文原计划的差异已在各节标注）

| 落点 | 新增 | 内容 |
|---|---|---|
| **新建** `deploy_config.py` | 12 | 监听地址 / TLS / OpenAI 身份 / 日志 / `context_manager` |
| **新建** `infer_config.py` | 7 | `infer_backend` `max_batch_size` `val_dataset_sample` `result_path` `write_batch_size` `metric` `reranker_use_activation` |
| `rollout_config.py` | 16 | 全部 `sglang_*`（规则 4） |
| `train_config.py` | 18 | 精度/性能 9 + 评估 5 + 损失 2 + 嵌套 2 |
| `logging_config.py` | 9 | swanlab 邮件 5 + `run_name` `logging_strategy` `logging_nan_inf_filter` `disable_tqdm` |
| `convert_config.py` | 7 | `merge_lora` `to_peft_format` `to_ollama` `to_cached_dataset` `template_mode` `commit_message` `exist_ok` |
| `checkpoint_config.py` | 6 | hub 全套（已去重，见下） |
| `distributed_config.py` | 6 | `fsdp_config` `ddp_broadcast_buffers` `ddp_bucket_cap_mb` `ddp_static_graph` `local_rank` |
| `model_config.py` | 4 | `model_kwargs` `external_plugins` `custom_register_path` `enable_npu_model_patch` |
| `quantize_config.py` | 3 | `quant_n_samples` `quant_batch_size` `group_size` |
| `generation_config.py` | 3 | `generation_config` `generation_max_length` `generation_num_beams` |
| `rlhf_config.py` | 1 | `reward_template` |
| `dataset_config.py` | 1 | `dataloader_drop_last` |

### 落点与原计划的 5 处差异

1. **`sampling_config.py` 那 8 项全部未迁** —— 逐项核对后全是改名或已在 recipe 签名，详见 1.7。这是本次最大的修正。
2. **hub 去重已定案**：`push_to_hub` / `hub_model_id` / `hub_private_repo` / `hub_strategy` / `hub_revision` / `hub_always_push` 全部归 `checkpoint_config.py` 一处；`convert_config.py` 只留 `commit_message`（提交信息不是凭证）。
3. **`fsdp_config` 改归 `distributed_config.py`**（紧邻已有的 `fsdp`），而非 `train_config.py`。
4. **`dataloader_drop_last` 改归 `dataset_config.py`**（dev 把 `dataloader_*` 都归在那里）。
5. **HF `log_level` 未迁**：与 `DeployConfig.log_level`（uvicorn 级别）重名但语义不同，dev 自闭环后 CLI 把各 config 平铺成一个参数命名空间时会碰。保留 deploy 那个（用户实际会传的），并在 `logging_config.py` 就地注释了原因。故 HF 实际迁 **37** 而非 38。

### 验收口径

| 验证 | 结果 |
|---|---|
| 16 个 config 均可无参构造 | 全部通过 |
| `typing.get_type_hints()` 逐类解析（探 import 缺失） | 全部通过。过程中抓到 `distributed_config` 缺 `Any/Dict/Union`、`train_config` 缺 `Any`，已补 |
| legacy-only 差集 | 114 → **56** |
| 剩余 56 项是否等于判定的不迁集 | 是，逐类完全对应 |
| ruff | 18 条全为 `I001`，且命中目录内**每个**文件（含未碰过的 `adapter_config` / `validate`）—— 既有目录级状况，新增规则类型为 0 |
| 行宽 | 改过的文件全部 ≤120 列 |

### HF 字段真伪校验（本次抳回一个错误）

所有要加的 HF 字段都用 `dataclasses.fields(Seq2SeqTrainingArguments)` 验过名字与默认值。过程中发现我曾凭印象写下的 **`eval_metric_prefix` 在 HF 中并不存在**，已删除。其余 28 项全部存在。

---

## 判定规则

1. **按类型迁移**：落点由字段语义决定（sglang 引擎参数 → `RolloutConfig`，日志 → `LoggingConfig`），不沿用 legacy 的 `*_args.py` 分文件结构。
2. **HF transformers 参数**：不全要，只迁常用的。legacy 通过继承 `Seq2SeqTrainingArguments` 免费获得全部 117 个，dev 需显式声明，故须逐个取舍。
3. **tuners 不要**：`boft` / `vera` / `reft` / `fourier` / `llamapro` / `lora_ga` / `adapter_*` 一律不迁。但 **galore、muon 等 optimizer 要保留**。
4. **lmdeploy 不迁**；**sglang 和 vLLM 都要支持**。
5. **wandb / swanlab 参数要保留**，tensorboard 同。注意这不是字段缺口而是**集成缺口** —— 见 1.5 后的说明与 2.3。
6. **UI 系列（app / webui）暂时不要**。
7. **merge_lora、sampling 参数要保留**。
8. **eval 功能暂时不迁移**。
9. **缺失的 config 建起来**：本轮新建 `DeployConfig`；eval 因规则 8 不建。
10. **日志/实验跟踪要真正支持**：`report_to`、swanlab、tensorboard 不只是有字段，还要接进链路（见 2.3）。
11. **后处理暂不处理**：推导 / 校验 / 副作用的迁移不在本轮范围（第四节仅作登记）。
12. **dev 自闭环**：去掉从 legacy Arguments 转换成 config 的路径，dev 侧不再依赖 `swift/arguments`（见 2.4）。

---

# 零、`swift/megatron/arguments/`（128 项，已全部迁入）

> **本表先前的全部口径只覆盖 `swift/arguments/`，遗漏了 `swift/megatron/arguments/`。**该目录有 6 个文件、335 个字段，其中 **247 项是 `swift/arguments/` 没有的独有字段**，再扣除 dev 已有的同名项，剩 **128 项既未迁移也未被任何文档记录**。

dev 侧代码其实早已自述这个缺口，见 `distributed_config.py:46-52`：

> `this is a MINIMAL subset of legacy MegatronArguments (megatron_args.py has 200+ fields). Only the parallelism sizes + the few high-frequency knobs below are wired into dev's build_model path today; the rest (fusion/fp8/mtp/muon/precision-aware-optimizer/vpp/...) are intentionally deferred.`

即：这是已知的待办，不是新发现的缺陷；但本文档之前从未把它计入对账，于是从“字段迁移已完成”的叙述里消失了。

## 0.1 已迁移：规则 3 / 5 要求的 18 项

这三组是规则 3 与规则 5 明确要求保留的，先前因扫描范围遗漏而被当成“不存在”，现已迁入。

| 组 | 数量 | 字段 | 落点 | 依据 |
|---|---|---|---|---|
| **muon** | 11 | `optimizer` + `muon_momentum` `muon_split_qkv` `muon_use_nesterov` `muon_scale_mode` `muon_fp32_matmul_prec` `muon_coefficient_type` `muon_num_ns_steps` `muon_tp_mode` `muon_extra_scale_factor` `muon_scalar_optimizer` | `train_config.py`（新增“Optimizer: Muon (Megatron-only)”区块） | 规则 3 |
| **邻接优化器项** | 2 | `sgd_momentum` `adam_eps` | 同上 | 同区块，一并带上 |
| **wandb** | 3 | `wandb_project` `wandb_exp_name` `wandb_log_unique_prompts` | `logging_config.py` | 规则 5 |
| **tensorboard** | 2 | `tensorboard_dir` `tensorboard_queue_size` | `logging_config.py` | 规则 5 |

两点必要的判断：

1. **`optimizer` 必须同时迁**。`megatron_args.py:500` 是 `optimizer: Literal['adam','sgd','muon','dist_muon'] = 'adam'` —— 它是 muon 的开关，不迁则那 10 个 `muon_*` 全部无法启用。它与 dev 已有的 `optim`（HF/torch 优化器名）**同位不同名也不同义**，两个 backend 各读其一，已在字段注释里写清。
2. **`wandb_project` 默认值改了**：legacy megatron 是 `'megatron-swift'`，这里取 `'ms-swift'`，与 dev 已有的 `swanlab_project` 保持一致 —— dev 的一次运行并不必然是 megatron。这是有意的偏离，不是照搬遗漏。

另外，legacy `_check_muon()`（`megatron_args.py:967`）含 4 条后处理，按规则 11 **本轮未迁**，在此登记：① 要求 `megatron-core>=0.16`；② `optimizer=='muon'` 时断言 `overlap_grad_reduce` / `overlap_param_gather` 均为假（否则要用 `dist_muon`）；③ 强制 `use_distributed_optimizer = False`；④ 派生别名 `muon_nesterov = muon_use_nesterov`（mcore 0.17 兼容）。这四条在 dev 目前无人执行。

## 0.2 次批：剩余 110 项已全部迁入

分组经脚本校验：**零重复、零遗漏，12 组合计 110 = 缺口集 110**。落点如下（合计 110 ✓）：

| 落点 | 数 | 来源组 |
|---|---|---|
| **新建 `moe_config.py`**（`MoEConfig`） | 13 | C |
| **新建 `megatron_config.py`**（`MegatronConfig`） | 28 | D 8 + J 9 + K 的引擎/初始化/GC/序列子集 11 |
| `train_config.py` | 26 | A 6 + B 8 + F 9 + K 3 |
| `distributed_config.py` | 20 | E 9 + G 6 + I 4 + K 1（`recompute_modules`） |
| `checkpoint_config.py` | 10 | H |
| `rlhf_config.py` | 10 | L |
| `model_config.py` | 3 | K |

### 两个新 Config 的边界

**`MoEConfig`**：路由 / 分发 / 容量 / 辅助损失。EP 的**尺寸**仍在 `DistributedConfig`（它说模型怎么摆），这里只说摆好之后怎么行为。

**`MegatronConfig`**：判据是“这个字段对 transformers backend 毫无意义”。包含算子融合、注意力实现、DSA/MHC/CSA、进程初始化、手动 GC、`megatron_extra_kwargs`。并行尺寸与通信重叠归 `DistributedConfig`，MoE 路由归 `MoEConfig`，两个 backend 都能用的留在共享 config。

全部默认值照搬 Megatron 现行值，**不动这两个 config 即复现当前行为**。

### A / B 的处理：沿用 dev 已有的 `clip_grad` 模式

实测：megatron 侧**只有 megatron 命名**（`lr` / `train_iters` / …），无任何 HF 名字（除 `num_train_epochs`，dev 已有），也没有转换代码 —— 两套命名在 legacy 里从不共存。

而 dev 的 `train_config.py` 是两个 backend 共用的，对此已有先例：`clip_grad`（megatron）与 `max_grad_norm`（HF）并存，`clip_grad` 保持 `Optional=None` 以区分“未设”，由 `resolve_max_grad_norm()` 单点折叠。故本批同样处理：

| megatron 拼写 | HF 对应物 | 处理 |
|---|---|---|
| `lr` | `learning_rate` | 别名，`Optional=None` |
| `train_iters` | `max_steps` | 别名，`Optional=None` |
| `micro_batch_size` | `per_device_train_batch_size` | 别名，`Optional=None`（legacy 默认 1）|
| `lr_warmup_fraction` | `warmup_ratio` | 别名，`Optional=None` |
| `lr_warmup_iters` | `warmup_steps` | 别名，`Optional=None`（legacy 默认 0）|
| `lr_decay_style` | `lr_scheduler_type` | **不完全等价**：`'WSD'` 无 HF 对应物，故保留完整 Literal |
| `global_batch_size` | `per_device × grad_accum × dp` | 方向相反（megatron 由它反推累加数），`Optional=None` |
| `lr_decay_iters` `lr_warmup_init` `lr_wsd_decay_*` `eval_iters` `finetune` `microbatch_group_size_per_vp_stage` | 无 | megatron 独有，照搬默认值 |

> **改了 3 个 legacy 默认值**：`micro_batch_size` 1→None、`lr_warmup_iters` 0→None、`global_batch_size` 16→None。不改就无法区分“用户显式传了 megatron 参数”与“默认值碰巧等于它”，折叠器也就写不对 —— 与 `clip_grad` 同理。

### 还没做的事

1. **折叠器不存在**。`clip_grad` 有 `resolve_max_grad_norm()`，但上表那 7 个别名**没有任何代码折叠它们**。同时传 `--lr` 和 `--learning_rate` 现在不会报错，也不会生效。属后处理，按规则 11 本轮不做。
2. **muon 的互斥校验现在可以写了**：`overlap_grad_reduce` / `overlap_param_gather` 已随 E 组进入 `DistributedConfig`，之前“写不出那条校验”的障碍已消除（但校验本身属后处理，未写）。
3. **`moe_aux_loss_coeff` 类型修了**：legacy 是 `List[float] = 0.`（注解与默认值矛盾），改为 `Union[float, List[float]] = 0.`。

## 0.3 次批 110 项的完整字段清单

| 组 | 数 | 字段 |
|---|---|---|
| **A 训练节拍** | 6 | `train_iters` `eval_iters` `global_batch_size` `micro_batch_size` `finetune` `microbatch_group_size_per_vp_stage` |
| **B 学习率调度** | 8 | `lr` `lr_decay_iters` `lr_decay_style` `lr_warmup_fraction` `lr_warmup_init` `lr_warmup_iters` `lr_wsd_decay_iters` `lr_wsd_decay_style` |
| **C MoE** | 13 | `moe_aux_loss_coeff` `moe_z_loss_coeff` `moe_router_load_balancing_type` `moe_router_dtype` `moe_token_dispatcher_type` `moe_token_drop_policy` `moe_expert_capacity_factor` `moe_pad_expert_input_to_capacity` `moe_grouped_gemm` `moe_permute_fusion` `moe_shared_expert_overlap` `moe_layer_recompute` `moe_enable_deepep` |
| **D 算子融合** | 8 | `apply_rope_fusion` `bias_activation_fusion` `bias_dropout_fusion` `cross_entropy_loss_fusion` `cross_entropy_fusion_impl` `gradient_accumulation_fusion` `masked_softmax_fusion` `apply_dsa_kernel_fusion` |
| **E 通信重叠** | 9 | `overlap_grad_reduce` `overlap_param_gather` `overlap_param_gather_with_optimizer_step` `overlap_p2p_comm` `batch_p2p_comm` `align_grad_reduce` `align_param_gather` `tp_comm_overlap` `nccl_comm_warmup` |
| **F 优化器精度/offload** | 9 | `use_precision_aware_optimizer` `main_params_dtype` `main_grads_dtype` `exp_avg_dtype` `exp_avg_sq_dtype` `accumulate_allreduce_grads_in_fp32` `optimizer_cpu_offload` `optimizer_offload_fraction` `optimizer_cuda_graph` |
| **G 流水线布局** | 6 | `virtual_pipeline_model_parallel_size` `pipeline_model_parallel_layout` `decoder_first_pipeline_num_layers` `decoder_last_pipeline_num_layers` `account_for_embedding_in_pipeline_split` `account_for_loss_in_pipeline_split` |
| **H checkpoint** | 10 | `async_save` `save_safetensors` `use_persistent_ckpt_worker` `no_load_optim` `no_load_rng` `no_save_optim` `no_save_rng` `dist_ckpt_optim_fully_reshardable` `distrib_optim_fully_reshardable_mem_efficient` `dist_ckpt_save_pre_mcore_014` |
| **I 并行策略** | 4 | `cp_comm_type` `cp_partition_mode` `expert_tensor_parallel_size` `data_parallel_sharding_strategy` |
| **J 注意力 / DSA / MHC / CSA** | 9 | `attention_backend` `attention_softmax_in_fp32` `apply_query_key_layer_scaling` `linear_decoupled_in_proj` `dsa_indexer_loss_coeff` `dsa_indexer_use_sparse_loss` `use_fused_mhc` `mhc_recompute_layer_num` `csa_dense_mode` |
| **K 初始化 / GC / 杂项** | 18 | `megatron_extra_kwargs` `skip_megatron_init` `perform_initialization` `use_cpu_initialization` `data_parallel_random_init` `te_rng_tracker` `manual_gc` `manual_gc_eval` `manual_gc_steps` `recompute_modules` `mlp_padding_free` `sequence_packing_scheduler` `calculate_per_token_loss` `apply_wd_to_qk_layernorm` `language_model_only` `mtp_shared_weights` `vit_attn_impl` `vit_gradient_checkpointing_kwargs` |
| **L megatron RLHF** | 10 | `mcore_ref_model` `mcore_ref_adapter` `calculate_KL` `f_divergence_type` `reference_free` `real_tau` `num_generations_eval` `router_replay_mode` `offload_bridge` `_teacher_use_disable_adapter` |

### 当时识别的四个难点与实际处理

| 难点 | 说明 |
|---|---|
| **A / B 与 dev 已有字段语义重叠** | 预判“这 14 项很可能是改名”—— 实测确认其中 **7 项确为别名**（见 0.2 的对应表）。处理：沿用 `clip_grad` 模式保留 megatron 拼写 + `Optional=None`，而非删除或直接堆入 |
| **E 与 muon 互斥** | 当时 `overlap_grad_reduce` / `overlap_param_gather` 不在 dev，写不出那条校验。现已随 E 组进入 `DistributedConfig`，障碍消除（校验本身属后处理，未写） |
| **K 的 `megatron_extra_kwargs`** | 已迁入 `MegatronConfig`。它的存在意味着未来 Megatron 新增的 flag 不必再逐个补字段 |
| **L 与 dev `RLHFConfig` 的关系** | 已查：10 项全为 megatron 路线特有，与已有 93 字段无同名。归入 `rlhf_config.py` 的“Megatron backend”区块，`mcore_ref_model` 与 `ref_model` 并存的理由已就地注释 |

## 0.4 关于默认值遗盖的原担心（已避开）

当初担心“把 110 个字段堆进 `DistributedConfig` 会引入第二个真相来源，并把 twinkle `MegatronModel` 的默认值遮盖成 None”（`distributed_config.py` 原注：“A None value means keep twinkle MegatronModel's own default”）。

实际做法从两个方向避开了它：

1. **默认值照搬 legacy megatron 原值**，而非一律改成 None —— 例如 `bias_activation_fusion=True`、`overlap_p2p_comm=True`、`moe_grouped_gemm=True` 都保留了 Megatron 的真实默认。不动新 config 即复现当前行为。例外只有那 3 个别名字段（见 0.2），它们必须是 None 才能区分“未设”。
2. **按类型拆成 `MoEConfig` / `MegatronConfig`**，而不是全堆进 `DistributedConfig`（规则 1）。

> 依然成立的保留：这些字段**目前无人读取**，所以“复现当前行为”目前是平凡成立的。真正的验证要等接线后做 bit-exact 对比。

---

## 对账口径

所有计数由脚本实测，不凭推算。口径：

- 用 `ast` 解析 `swift/arguments/**.py` 与 `swift/dev/config/**.py` 下每个 `ClassDef` 的 `AnnAssign` 字段名，取并集后做差集。
- HF 侧用 `dataclasses.fields(transformers.Seq2SeqTrainingArguments)` 取真实字段名，避免手抄。
- **口径局限（重要）**：这是**同名差集**。若 dev 把某字段改了名，会被误判成缺口。已确认一例改名（见下），其余待迁移项**未逐个到 dev 侧查找语义等价物**，是本表最主要的不确定来源。

| 项 | 数值 |
|---|---|
| legacy 类数 / 本地声明字段数 | 26 / 327 |
| legacy 经继承获得的 HF 字段 | +101（`Seq2SeqTrainingArguments` 共 117 个可传） |
| **legacy 实际参数面** | **428** |
| dev 类数 / 声明字段数 | 14 / 410 |
| 同名交集 | 213 |
| legacy 独有（缺口） | **114** |
| HF 参数未被 dev 覆盖 | **63**（dev 已覆盖 54/117） |
| dev 真正新增（非 HF 重声明） | 153 |

**待分类总数 = 114 + 63 = 177。**

> 两边**不是包含关系，是重新划分**：dev 有 153 个 legacy 没有的字段。另 44 个"dev 独有"实为 HF 参数的显式重声明（legacy 靠继承拿到）。

### 重名字段说明

按 legacy 文件求和会得到 117，比去重后的 114 多 3，原因：

- `verbose` 出现在 `deploy_args.py` / `eval_args.py` / `app_args.py`（3 次 → 计 1）
- `lang` 出现在 `app_args.py` / `webui_args.py`（2 次 → 计 1）

处置：`verbose` 因 deploy/eval 需要而**迁移**（不随 UI 砍掉）；`lang` 仅 UI 使用，**不迁**。

## 列含义

`字段`：legacy 字段名 ｜ `落点`：dev 目标 Config ｜ `结论`：migrate / drop ｜ `依据`：对应上面哪条判定规则或实测结论。

---

# 一、不迁移（56 项，实测）

> 这 56 项就是迁移后剩下的全部 legacy-only 字段，与脚本差集逐项对应：
> tuner 25 + sampling 8 + eval 5 + lmdeploy 5 + app 4 + webui 3 + `bf16`/`fp16` 2 + `ignore_args_error`/`use_swift_lora` 2 + `response_length`/`seq_kd` 2 = **56** ✓

## 1.1 tuner 类型 — 25 项（规则 3）

已实测确认：`tuner_args.py` 的 25 个缺口中**不含任何 optimizer 类字段**，故可整块砍掉，不会误伤 galore / muon。

| 组 | 数量 | 字段 |
|---|---|---|
| boft | 4 | `boft_block_num` `boft_block_size` `boft_dropout` `boft_n_butterfly_factor` |
| vera | 4 | `vera_rank` `vera_dropout` `vera_d_initial` `vera_projection_prng_key` |
| reft | 5 | `reft_rank` `reft_layers` `reft_layer_key` `reft_args` `reft_intervention_type` |
| lora_ga | 6 | `lora_ga_batch_size` `lora_ga_direction` `lora_ga_iters` `lora_ga_max_length` `lora_ga_scale` `lora_ga_stable_gamma` |
| fourier | 2 | `fourier_n_frequency` `fourier_scaling` |
| llamapro | 2 | `llamapro_num_groups` `llamapro_num_new_blocks` |
| adapter | 2 | `adapter_act` `adapter_length` |

**galore / muon 的实测状态**（规则 3 要求保留，结论是已满足）：

| 项 | legacy 声明 | dev 声明 | 结论 |
|---|---|---|---|
| `galore*` | 0 | 15 | 无需迁移，dev 更完整（`adapter_config.py` + `validate.py`） |
| `muon` | 0 | 有（`distributed_config.py`） | dev 新增能力，legacy 无 |
| `lorap*` | 1 | 2 | 无 legacy 独有项 |

## 1.2 lmdeploy — 5 项（规则 4）

`lmdeploy_cache_max_entry_count` `lmdeploy_quant_policy` `lmdeploy_session_len` `lmdeploy_tp` `lmdeploy_vision_batch_size`

## 1.3 UI — 7 项（规则 6）

| 来源 | 字段 |
|---|---|
| `app_args.py` | `base_url` `is_multimodal` `studio_title` `lang` |
| `webui_args.py` | `server_name` `server_port` `share` |

## 1.4 已改名，非缺口 — 2 项（实测）

| legacy 字段 | dev 等价物 | 位置 |
|---|---|---|
| `bf16` `fp16` | `torch_dtype: Literal['bfloat16','float16','float32',None]` | `model_config.py:16` |

## 1.5 HF 非常用 — 25 项（规则 2）

**这一格是我的判断，需重点复核。**

| 组 | 字段 |
|---|---|
| 流程开关 | `do_train` `do_predict` `debug` `use_cpu` `skip_memory_metrics` `sortish_sampler` |
| trackio | `trackio_bucket_id` `trackio_space_id` `trackio_static_space_id` `project` |
| 评估细节 | `eval_do_concat_batches` `eval_use_gather_object` `include_for_metrics` `batch_eval_metrics` |
| 日志分级 | `log_level_replica` `log_on_each_node` |
| 其他 | `length_column_name` `optim_target_modules` `parallelism_config` `enable_jit_checkpoint` `train_sampling_strategy` `restore_callback_states_from_checkpoint` `include_num_input_tokens_seen` |

### `liger_kernel_config` / `parallelism_config` / `accelerator_config` 的来龙去脉

这三个是 HF 的字段，容易和 dev 已有的东西混淆，单独说明（HF 帮助文本实测摘录）：

| HF 字段 | 类型与含义 | legacy 有什么 | dev 有什么 | 结论 |
|---|---|---|---|---|
| `use_liger_kernel` | 开关 | 有（经 HF 继承，`rlhf_args.py` 用它做了 7 处互斥校验） | **已有** `train_config.py:61` + `validate.py:377` | 已迁移，无需处理 |
| `liger_kernel_config` | `dict[str, bool]`，作为 kwargs 传给 `_apply_liger_kernel_to_instance()`，即**逐 op 指定 patch 哪些算子** | 仅经继承，无本地声明 | 无 | 建议迁移：twinkle 侧 liger 是 op 级注册而非整模型 patch，这个 dict 正是 op 级粒度的入口 |
| `accelerator_config` | `dict \| str \| None`，Accelerate 集成配置（JSON 路径或 dict） | 仅经继承 | 无 | 建议迁移（归 2.2 嵌套配置） |
| `parallelism_config` | `accelerate.parallelism_config.ParallelismConfig`，需 Accelerate ≥ 1.10.1 | 仅经继承，**legacy 无任何本地并行字段与之对应** | 无此字段，但 `distributed_config.py` 用 24 个**显式**并行字段自建模型 | **不迁移**：它是 Accelerate 的另一套并行描述机制，与 dev 自己的显式并行建模重叠，同时存在会出现两个真相来源 |

## 1.6 eval — 5 项（规则 8）

eval 功能暂时不迁移。dev 侧目前连 eval recipe 都没有。

`eval_backend` `eval_url` `eval_output_dir` `eval_num_proc` `local_dataset`

> `verbose` 原本被 eval 与 deploy 共用，deploy 仍需要它，故 `verbose` 已随 `deploy_config.py` 迁入。

## 1.6 UI — 7 项（规则 6）

`app_args.py`：`base_url` `is_multimodal` `lang` `studio_title`；`webui_args.py`：`server_name` `server_port` `share`。

## 1.7 改名 / 已废弃 / 已在别处 — 12 项（本次核对新增）

这一类**不是缺口**，直接加会造出两个同义字段。原计划把它们归入待迁移，是错的；文档第四节第 1 项预告的风险就是这个。

| legacy 字段 | dev 里的实际归宿 | 依据 |
|---|---|---|
| `num_sampling_batch_size` | `SamplingConfig.batch_size` | 同义：每轮采样的批大小 |
| `num_sampling_batches` | `SamplingConfig.max_batches` | 同义：总批数上限 |
| `prm_threshold` | `SamplingConfig.reward_threshold` | 同义：低于阈值则过滤 |
| `orm_model` | `SamplingConfig.reward_funcs` | 架构性替代：dev 用 registry 名/callable 列表而非模型 ID |
| `prm_model` | `SamplingConfig.prm_funcs` | 同上 |
| `sampler_engine` | `run_sampling(backend=...)` | 已是 recipe 函数签名参数（'vllm'/'sglang'/'transformers'/'client'） |
| `engine_kwargs` | `run_sampling(engine_args=...)` | 同上 |
| `sampler_type` | `backend='client'` | legacy 的 'distill' 在 dev 用教师端 backend 表达 |
| `response_length` | `RLHFConfig.max_completion_length` | **legacy 源码自注** `# compat. use max_completion_length instead` |
| `seq_kd` | 无 | **legacy 源码自注** `# Deprecated` |
| `use_swift_lora` | `TunerConfig.tuner_backend` | **legacy 源码自注** `# True for using tuner_backend == swift` |
| `ignore_args_error` | — | legacy 参数解析的容错开关（notebook 兼容），属 CLI 行为而非配置 |

> 前 8 项计入 sampling、后 4 项计入 rlhf/base，不重复计数。

---

# 二、待迁移（已完成 —— 以下为原计划，保留作对账依据）

> 实际落地见文首的「本轮实际落地」表。本节的分组是执行前的计划，其中 2.1 的 sampling 8 项实为改名（已移至 1.7），hub / `fsdp_config` / `dataloader_drop_last` 的落点也做了调整。

## 2.1 legacy 本地声明 — 原计 70 项，实迁 58 项

| 落点 Config | 数量 | 字段 | 依据 |
|---|---|---|---|
| `rollout_config.py` | 16 | `sglang_tp_size` `sglang_dp_size` `sglang_pp_size` `sglang_ep_size` `sglang_context_length` `sglang_mem_fraction_static` `sglang_kv_cache_dtype` `sglang_quantization` `sglang_disable_cuda_graph` `sglang_disable_custom_all_reduce` `sglang_enable_dp_attention` `sglang_enable_ep_moe` `sglang_speculative_algorithm` `sglang_speculative_eagle_topk` `sglang_speculative_num_draft_tokens` `sglang_speculative_num_steps` | 规则 4 |
| `convert_config.py` | 12 | hub：`push_to_hub` `hub_model_id` `hub_private_repo` `commit_message` `exist_ok` ｜ 导出目标：`to_ollama` `to_peft_format` `to_cached_dataset` `template_mode` ｜ 量化：`quant_batch_size` `quant_n_samples` `group_size` | — |
| **新建 `deploy_config.py`** | 12 | `host` `port` `api_key` `ssl_keyfile` `ssl_certfile` `served_model_name` `owned_by` `max_logprobs` `log_interval` `log_level` `verbose` `context_manager` | 规则 9 |
| `sampling_config.py` | 8 | `sampler_engine` `sampler_type` `orm_model` `prm_model` `prm_threshold` `engine_kwargs` `num_sampling_batches` `num_sampling_batch_size` | 规则 7 |
| `generation_config.py` 或新 infer config | 7 | `infer_backend` `max_batch_size` `metric` `result_path` `val_dataset_sample` `write_batch_size` `reranker_use_activation` | — |
| `model_config.py` | 6 | `model_kwargs` `external_plugins` `custom_register_path` `use_swift_lora` `enable_npu_model_patch` `ignore_args_error` | — |
| `logging_config.py` | 5 | `swanlab_smtp_server` `swanlab_smtp_port` `swanlab_sender_email` `swanlab_receiver_email` `swanlab_email_language` | 规则 5 |
| `rlhf_config.py` | 3 | `response_length` `reward_template` `seq_kd` | — |
| `adapter_config.py` | 1 | `merge_lora` | 规则 7 |

> 按落点求和即 **70**（eval 已按规则 8 移出，`verbose` 现在只属 deploy，不再重复）。
>
> swanlab 只缺邮件通知这 5 项：dev 的 `logging_config.py` 已有 `report_to` `swanlab_project` `swanlab_workspace` `swanlab_exp_name` `swanlab_mode` `swanlab_token` `swanlab_secret` `swanlab_webhook_url` `swanlab_notification_method` 共 9 个。wandb 两边都无独立字段，走 HF 的 `report_to`。**但这些字段目前是死的 —— 见 2.3。**

## 2.2 HF 常用参数 — 38 项（规则 2）

| 落点 Config | 数量 | 字段 |
|---|---|---|
| `train_config.py`（精度/性能） | 9 | `bf16_full_eval` `fp16_full_eval` `tf32` `torch_compile` `torch_compile_backend` `torch_compile_mode` `torch_empty_cache_steps` `auto_find_batch_size` `use_cache` |
| `checkpoint_config.py`（Hub） | 6 | `push_to_hub` `hub_model_id` `hub_strategy` `hub_private_repo` `hub_revision` `hub_always_push` |
| `train_config.py`（评估） | 5 | `do_eval` `eval_accumulation_steps` `eval_delay` `load_best_model_at_end` `prediction_loss_only` |
| `logging_config.py` | 5 | `run_name` `log_level` `disable_tqdm` `logging_strategy` `logging_nan_inf_filter` |
| `distributed_config.py` | 5 | `ddp_broadcast_buffers` `ddp_bucket_cap_mb` `ddp_static_graph` `dataloader_drop_last` `local_rank` |
| 嵌套配置 | 3 | `fsdp_config` `accelerator_config` `liger_kernel_config` |
| `generation_config.py` | 3 | `generation_config` `generation_max_length` `generation_num_beams` |
| `train_config.py`（损失） | 2 | `label_names` `label_smoothing_factor` |

> 嵌套配置这 3 项已根据规则 2 重新核定：`liger_kernel_config` 与 `accelerator_config` 保留；`parallelism_config` 改为**不迁**（见 1.5 后的说明）。故本行实为 2 项 + `fsdp_config`。

## 2.3 日志/实验跟踪：字段已在，集成未做（规则 10）

这不是字段缺口，而是集成缺口。实测现状：

| 事实 | 证据 |
|---|---|
| `LoggingConfig` 有 12 个字段，含 `report_to: List[str] = ['tensorboard']` 与 8 个 swanlab 字段 | `logging_config.py` |
| 类上方标注 `# TODO: integrate it` | `logging_config.py` |
| 全仓仅 `config/__init__.py` 引用它（import + `__all__`），**无任何 recipe / builder / validate 消费** | grep `LoggingConfig\|logging_config` 实测 |
| `validate_configs()` 签名不收 `logging_config` | 参数仅 model / template / dataset / train / distributed / checkpoint / tuner |

工作项：

1. 把 `logging_config` 接入 `validate_configs()` 签名与各 recipe 的调用点。
2. 在 builder / trainer 侧真正消费 `report_to`（tensorboard / swanlab / wandb 三路）。
3. 迁入 2.1 里那 5 个 swanlab 邮件通知字段。
4. 移除 `# TODO: integrate it`。

## 2.4 dev 自闭环：拆掉 legacy 转换桥（规则 12）

实测现状：dev 的三个 CLI 入口**仍在构造 legacy Arguments** 再转成 dev Config：

| 入口 | legacy 引用处数 | dev config 引用处数 | 转换桥 |
|---|---|---|---|
| `dev/cli/export.py` | 10 | 9 | `export_args_to_configs(args: 'ExportArguments') -> dict` |
| `dev/cli/megatron.py` | 20 | 11 | 待确认 |
| `dev/cli/sft.py` | 10 | 10 | 待确认 |

另外 `swift/megatron/convert.py` 也引用 `swift.arguments`（2 处）。

工作项：dev CLI 自行完成命令行解析 → 直接构造 Config，删除 `export_args_to_configs` 类转换函数与对 `swift/arguments` 的 import。

> 注意依赖方向：这一步完成后，legacy 侧的 (A) 推导与 (C) 副作用就**不再会被执行** —— 而根据规则 11 它们本轮不迁。两者合起来意味着：**拆桥不能先于后处理迁移完成**，否则会丢掉环境变量设置、分布式初始化、模型下载等 27 处行为。详见第四节第 2 项。

---

# 三、已定案与残留决策

## 已定案

| 事项 | 结论 |
|---|---|
| deploy 的 12 项 | **本轮新建 `deploy_config.py`**（规则 9） |
| eval 的 5 项 | **不迁**（规则 8）；`verbose` 随 deploy 保留 |
| `parallelism_config` | **不迁** —— 与 dev 自己的显式并行建模重叠，同时存在会出现两个真相来源 |
| `liger_kernel_config` / `accelerator_config` | **迁**（归 2.2 嵌套配置） |
| `use_liger_kernel` | 已在 dev，无需处理 |
| 日志/实验跟踪 | 字段已在但未接入，**集成列为工作项**（见 2.3） |
| 后处理（推导/校验/副作用） | **本轮不处理**（规则 11），实测结论仅登记于第四节 |
| legacy 转换桥 | **拆除**，dev 自闭环（规则 12，见 2.4） |

## 残留决策

| # | 事项 | 需要的决定 |
|---|---|---|
| 1 | **hub 参数重复** | `push_to_hub` / `hub_model_id` / `hub_private_repo` 同时出现在 2.1（`convert_config`）和 2.2（`checkpoint_config`）。**必须只保留一处**，否则会出现两个语义相同的字段 |
| 2 | **HF 其余划线** | 除上述已定案的三个嵌套配置外，36/26 的切分仍为我的判断，待复核 |
| 3 | **拆桥与后处理的先后** | 规则 12（拆桥）与规则 11（后处理不迁）相互冲突 —— 见第四节第 2 项，需定排序 |

---

# 四、已知未验证项

| # | 项 | 说明 |
|---|---|---|
| 1 | **同名差集的固有局限（风险已部分兑现）** | 本轮逐项核对时找出 **12 项假缺口**（见 1.7），其中 3 项由 legacy 源码自带注释坐实。另已确认 `bf16/fp16 → torch_dtype`。<br>**仍未完全消除**：已迁入的 58 项中，我只对有可疑同义字段的那几组做了逐项比对（sampling / rlhf / base / hub）；sglang 16 项与 deploy 12 项因 dev 侧完全无对应模块而直接判为缺口，未逐个反向搜查。 |
| 2 | **后处理：推导/校验已迁，副作用待迁** | 本轮已把 legacy `__post_init__` 里的 **纯推导 (a)** 与 **纯校验 (b)** 两类迁入 `process.py` / `validate.py`（见第五节）。仍未迁的是 **(c) 类副作用**：legacy 侧尚有 **27 处**（环境变量写入 6、分布式初始化 3、下载/登录 4、版本探测 2 等），它们需要真实 runtime，归建模处而不是一遍 dataclass 扫描。<br>**冲突点依旧**：拆桥（规则 12）不能先于这 27 处副作用的安项，否则这类丢失不报错。 |
| 3 | **(A) 类参数推导无归属** | dev/config 刻意做成纯数据类，故 `model_type → template`、`task_type`、`learning_rate` 默认值等推导在 dev 侧**目前没有落点**。需先定归属（`dev/builders/` 还是 CLI 映射层）才能开始搬。 |
| 4 | **legacy 目前仍是活代码** | `dev/cli/` 三个入口（export / megatron / sft）**仍在构造 legacy Arguments** 再转成 dev Config，故 (A)(C) 类逻辑今天照常执行，现在没有正确性 bug。规则 12 要做的正是拆掉这层。 |
| 5 | **新增字段尚无消费方** | 本轮只把字段放进 dataclass，**没有任何 builder / trainer / recipe 读取它们**。两个新 config（`DeployConfig` / `InferConfig`）也未接入 `validate_configs()` 签名。写入无效是预期的 —— 接线是下一步的工作。 |
| 6 | **`quantize_config` / `logging_config` / `generation_config` 仍挂着 `# TODO: integrate it`** | 这三个类本轮都加了字段，但那句 TODO 依然成立，没有因为字段变多而自动解决。 |

---

# 五、后处理迁移（`__post_init__` → `process.py` / `validate.py`）

> legacy 把推导、校验、副作用全混在 `__post_init__` 里。dev 拆成三处：`process.py` **只写推导值**，`validate.py` **只读只拒**，副作用归建模时。调用顺序固定：先 `process_configs()` 后 `validate_configs()`（校验针对已推导的值写）。

## 5.1 推导已迁入 `process.py`（13 项）

均为“只填默认/只规范化、不覆盖显式值”，与已有的 `_fold_megatron_aliases` / `_derive_*` 同风格：

| 新函数 | 做什么 | legacy 出处 | 涉及 Config 字段 |
|---|---|---|---|
| `_coerce_mrl_dims` | `mrl_dims` 的 key/value 转 `{int: float}` | `megatron_args.py:842-844` | `TrainConfig.mrl_dims` |
| `_derive_vit_gradient_checkpointing` | `vit_gradient_checkpointing = not freeze_vit` | `sft_args.py:211-212` `megatron_args.py:806-807` | `TrainConfig` × `TunerConfig.freeze_vit` |
| `_derive_packing_length` | `packing` 时 `packing_length` 默认 `max_length` | `base_args.py:198-199` | `DatasetConfig` × `TemplateConfig.max_length` |
| `_derive_split_dataset_ratio` | 有 val_dataset / streaming 则 ratio→0 | `data_args.py:110-116` | `DatasetConfig` |
| `_derive_eval_schedule` | 无验证集→'no'；eval_strategy←save_strategy；eval_steps←save_steps | `sft_args.py::_init_eval_strategy` + `:231-232` | `TrainConfig` × `CheckpointConfig` × `DatasetConfig` |
| `_derive_streaming_dataloader_workers` | streaming 下 worker 限 1 | `megatron_base_args.py:54-57` | `DatasetConfig` |
| `_derive_bnb_compute_dtype` | bnb 4bit compute dtype 从 `torch_dtype` 推 | `quant_args.py:116-122` | `ModelConfig.torch_dtype` × `QuantizeConfig` |
| `_derive_rlhf_task_type` | `rlhf_type=rm` → `seq_cls` + `num_labels=1` | `rlhf_args.py::_init_rm` | `ModelConfig` × `RLHFConfig` |
| `_derive_rlhf_beta` | `beta` 默认按 rlhf_type（grpo .04/gkd .5/simpo 2/其余 .1） | `_set_default` `_init_grpo` `_init_simpo` | `RLHFConfig` |
| `_derive_rlhf_ref_model` | 适用算法且 full 时 `ref_model←model`；grpo+beta0 清空 | `rlhf_args.py:289-297` | `RLHFConfig` × `ModelConfig` × `TunerConfig` |
| `_derive_grpo_reward_defaults` | `scale_rewards`/`kl_in_reward` 按 `advantage_estimator` | `rlhf_args.py::_init_grpo` | `RLHFConfig` |
| `_derive_best_model_metric` | metric 与 greater_is_better 按任务推 | `sft_args.py::_init_metric_for_best_model` `megatron_args.py:862-865` | `TrainConfig` × `RLHFConfig` |
| `_normalize_recompute_granularity` | 字符串 `'none'` → `None` | `megatron_args.py:797-798` | `DistributedConfig` |

新增形参：`process_configs()` 加 `quantize_config`；`process_configs` 已在 `config/__init__.py` 导出（之前只定义未导出）。

## 5.2 校验已迁入 `validate.py`（8 项）

| 新函数 | 做什么 | legacy 出处 |
|---|---|---|
| `_check_selective_recompute` | selective + recompute_method → raise | `megatron_args.py:799-800` |
| `_check_pipeline_decoder_layers` | pp==1 但设 decoder_first/last → raise | `megatron_args.py:832-835` |
| `_check_tp_comm_overlap` | tp_comm_overlap 需 sequence_parallel | `megatron_args.py:896-898` |
| `_check_sequence_parallel_tp` | sequence_parallel 需 tp>1（**改为 raise**，不像 legacy 静默置假） | `megatron_args.py:890-891` |
| `_check_save_total_limit` | 与 async_save 互斥；megatron 下需 >=2 | `megatron_args.py:857-861` |
| `_check_rlhf_ref_model` | CPO/ORPO/LoRA 传 ref_model → raise（推导的一半在 process） | `rlhf_args.py:297-298` |
| `_check_rlhf_padding_free` | 非 grpo/dpo/kto/gkd 不支持 padding_free/packing | `rlhf_args.py::_check_padding_free` |
| `_check_rlhf_sequence_parallel` | 非 grpo/dpo 不支持 sequence_parallel_size>1 | `rlhf_args.py::_check_sequence_parallel` |

新增形参：`validate_configs()` 尾部加可选 `rlhf_config`（默认 None，现有 recipe 调用点不变）。

## 5.3 故意不迁（与 dev 哲学冲突或缺判据）

| 项 | 原因 |
|---|---|
| `learning_rate` 按 tuner 推 1e-5/1e-4 | dev 的 `learning_rate` 默认是定值 1e-5（无 None 哨兵），无法区分“未设”与“显式 1e-5”；`test_optimizer_config.py` 已将此记为故意 break。 |
| ppo/gkd 强置 `padding_side='left'` | 无 None 哨兵，会静默覆盖用户显式 'right'；违 dev “不静默降级”原则。 |
| rlhf `loss_scale` 默认 | 依赖 `model_meta.is_multimodal`（需加载模型，属 (c) 类）。 |
| `apply_wd_to_qk_layernorm` 限 qwen3 变体 | 需已解析的 `model_type`，dev config 层 `model_type` 常为 None，会误报。 |
| 27 处环境变量/分布式初始化/下载/插件导入 | (c) 类副作用，归建模处（见四.2）。 |

## 5.4 验收

| 验证 | 结果 |
|---|---|
| `process_configs` + `validate_configs` 默认链可跑、幂等（跑两次同结果） | 通过 |
| 新推导/校验逐个实例验证（packing_length / split_ratio / rm / grpo / bnb / 各 raise） | 通过 |
| ruff（E/F/W） | 0；仅余 `I001`（目录级旧状况，`validate.py` 在 HEAD 即已 2 条），新增规则类型 0；`__init__.py` 保持全通过 |
| 行宽 | 改过的文件均 ≤120 列 |
| 已有配置测试（test_optimizer_config / test_megatron_cli_mapping / …） | 仅余 `twinkle.*` ModuleNotFound 与早批字段新增导致的 CLI 映射漂移（与本次无关） |

## 6. RL / RLHF recipe 迁移（`swift/dev/recipe/run_grpo|run_dpo|run_gkd|run_ppo`）

把 legacy 支持的全部 `rlhf_type` 组装为 dev recipe，colocate（同卡 IPC）与异构（分离卡 NCCL/HCCL）皆支持。入口按算法族拆分：

| recipe | 覆盖 `rlhf_type` | 形态 | rollout / 参考 |
|---|---|---|---|
| `run_grpo` | grpo（及 GSPO/RLOO/REINFORCE++ 变体经 `RLHFConfig`） | 在线 RL + 真 weight-sync | `vLLMSampler` + `CheckpointEngineManager`，old_logps 取自 `sequence.logprobs` |
| `run_dpo` | dpo / kto / cpo / orpo / simpo / rm | 离线偏好（无 rollout） | dpo/kto 参考：LoRA 走 `disable_lora`、full 载冻结 `ref_model`；cpo/orpo/simpo/rm 无参考 |
| `run_gkd` | gkd | 在线蒸馏 | student 用 `model.generate`（自身当前权重，**无需 weight-sync**）+ 冻结 teacher `forward_only(return_logits=True)` |
| `run_ppo` | ppo | 在线 RL | 复用 run_grpo 的 rollout；策略=GRPOLoss 裁剪、critic=seq_cls num_labels=1 头经 `task='value'` 出 per-token value + `PPOValueLoss`，per-token GAE |

### 6.1 rollout 后端选择（vLLMSampler vs RolloutEngine）

在线 RL 权重同步只对带 `CheckpointEngineMixin` 的 Ray-actor 采样器可行。故 `run_grpo`/`run_ppo` 的 rollout 后端是 twinkle `vLLMSampler`（`SamplerRollout` 包装），**不是** `swift.dev.rollout.RolloutEngine`（裸 `GRPOVllmEngine`，无同步能力，仅留作 `grpo.py` 本地无同步 smoke）。`GRPOLoop._rollout_step` 每步前若 rollout 暴露 `sync_weights` 即调用（generate 后调 `finish_generate`）；本地 smoke 无此 hook，自动跳过。

### 6.2 colocate / 异构判定（`plan_rl_device_groups`，纯函数）

由 `RolloutConfig.vllm_mode` 决定，`DistributedConfig.nproc_per_node` 语义为**训练器** GPU 数：
- `'colocate'`：训练器与采样器共用同一 `model` DeviceGroup（两个 remote_class 角色落在同批 GPU、rank 空间独立，无需改 placement）→ `CheckpointEngineManager(colocate=True)`（IPC）。recipe 执行显存调度：`sampler.wake_up(['weights'])→sync→model.offload_to_cpu→sampler.wake_up()→generate→sampler.sleep()→model.reload_to_gpu`。
- `'server'`/默认：`model` 组 `[0,M)` + 不相交 `sampler` 组 `[M,M+S)` → `CheckpointEngineManager(colocate=False)`（NCCL/HCCL）。镜像 `twinkle/tests/sampler/test_weight_sync.py`。

### 6.3 build_model transformers 缺口补齐（Subsystem D）

legacy 仅 Megatron 分支设 `remote_group='model'`+`device_mesh`。`_build_transformers_model` 新增 `_apply_ray_placement`：`mode!='local'` 时按 `DeviceMesh.from_sizes(world_size=nproc, dp_size=nproc)`（纯 DP）设 `remote_group='model'`。并给 `TransformersModel`/`AccelerateStrategy` 补 `offload_to_cpu`/`reload_to_gpu`（原仅 `MegatronModel` 有），供 colocate 显存调度。

### 6.4 loss 装配来源（`configure_rlhf_loss` + `RewardLoss` + `PPOValueLoss`）

`swift/dev/loss/configure.py::configure_rlhf_loss(model, rlhf_config)` 按 `rlhf_type` 映射 twinkle loss（`_RLHF_LOSS_NAME`）：grpo/dpo/cpo/orpo/simpo/gkd 已在 twinkle 现存；kto 复用 DPO 族 `loss_type='kto_pair'`（成对近似，非 legacy 非成对 KTO，已记为简化）；rm 用**新增** `twinkle/loss/reward.py::RewardLoss`（pairwise Bradley-Terry，`center_rewards_coefficient`），配 `task_type='seq_cls', num_labels=1` 头；**ppo 的策略 loss** 就是同一裁剪 surrogate → 映射 `GRPOLoss`（`epsilon=cliprange`，KL 在 loop 的 reward 整形里施加、loss 内 beta=0 避免双计）。ppo 的 critic 用**新增** `twinkle/loss/value.py::PPOValueLoss`（裁剪 value 回归，`cliprange_value`/`vf_coef`），由 `configure_ppo_value_loss(value_model, rlhf_config)` 单独设在 value model 上。仅转发各算法真正读取的字段，其余留 twinkle 默认；`beta` 缺省已由 `process.py::_derive_rlhf_beta` 填好。

### 6.5 偏好数据管线（Subsystem B）

`run_dpo` 用 `build_dataset(encode=False)` 取原始行，`PreferenceLoop` 逐 micro-batch 经 `template.encode`（`mode='rlhf'`，kto 为 `'kto'`，rm 走 `task_type='seq_cls'` 自动丢 labels）拆前缀 `chosen_*`/`rejected_*` 为两个 InputFeature，并**交错** `[chosen_1, rejected_1, chosen_2, rejected_2, ...]`——正是 twinkle DPO 族 `_split_chosen_rejected`（偶/奇索引）所需布局，且每 micro 序列数恒为偶且相等，保证 GA 正确。

### 6.6 参考模型 logps（Subsystem C）

dpo/kto：LoRA 用 `model.forward_only(inputs, disable_lora=True)` 取 `outputs['logps']`（不额外载模型）；full 载冻结 `ref_model`（`process.py::_derive_rlhf_ref_model` 默认取 policy 初值）`forward_only`。逐 batch 作 `ref_logps=` 注入偏好 loss。cpo/orpo/simpo/rm 无参考。为单进程 `mode='local'` 路径（driver 内 in-process），ray/megatron 偏好参考不在本 recipe 范围。

### 6.7 PPO（per-token GAE，含可训练 critic，双后端对称）

critic 复用 **seq_cls `num_labels=1` 头**（两后端同构：mcore-bridge `OutputLayerLinear` / HF `AutoModelForSequenceClassification`），但用**新增 `task='value'`** 前向保留其 **per-token** 输出（跳过 seq_cls 的末 token 池化），故每个 token 都有一个 value `V(s_t)`，经 `outputs['logits']`（`[B,T]`）返回。两后端对称实现：transformers 侧新增 `twinkle/patch/transformers_value.py::TransformersValuePatch`（hook seq_cls `score` 头，surface 池化前的 per-token 分数）；megatron 侧 `forward_step` 新增 `task=='value'` 分支（CP 重建后跳过末 token pick，per-token 走既有 logits 通道）。`_resolve_task_context`（两文件）均登记 `'value'`。

每步：reward model 标量在**末 token**计入，每个 response token 的 reward `r_t = -kl_coef·(logp_t − ref_logp_t)`（per-token KL(policy‖ref)）；`twinkle/advantage/gae.py::GAEAdvantage`（`GRPOAdvantage`/`RLOOAdvantage` 的同侪）沿 response 反向走 `gamma`/`lam` 出 per-token advantages（喂裁剪策略 surrogate）与 returns（喂裁剪 value loss）。advantages/old_logps/returns/old_values 每 rollout 只算一次，重用 `num_ppo_epochs` 次。策略 loss=`GRPOLoss`（`epsilon=cliprange`，KL 在 reward 整形施加、loss 内 beta=0 避免双计），critic loss=**新增** `twinkle/loss/value.py::PPOValueLoss`（裁剪 value 回归，`cliprange_value`/`vf_coef`，per-token 目标经 label mask 散射到 response 位），由 `configure_ppo_value_loss` 单独设在 value model 上。rollout/weight-sync/colocate 与 run_grpo 同。**新增 twinkle 组件**：`PPOValueLoss`（注册 `'ppo_value'`）、`GAEAdvantage`、`TransformersValuePatch`、两后端 `task='value'`。build_model 不再有 `value_head` 特判——critic 走普通 seq_cls 构建路径。

### 6.8 测试分层

- 便宜层（`swift/dev/tests/test_rl_recipes.py`，进 CI，无 Ray/vLLM/GPU）：`plan_rl_device_groups` colocate/异构/校验；`PreferenceLoop` 交错布局+前缀剥离+RM 无 labels+缺边报错；`GKDLoop` prompt 窗口回绕；`configure_rlhf_loss` 类型→loss 映射（`importorskip('twinkle.loss')`，twinkle 缺失时跳过）。
- 重型层（`@pytest.mark.slow`，多卡门禁，镜像 `twinkle/tests/sampler/test_weight_sync.py`，手动跑）：异构/colocate weight-sync 前后采样输出变化 + 参数更新；DPO/GKD e2e loss 下降 smoke。

### 6.9 已知简化

| 项 | 简化 |
|---|---|
| KTO | 复用 DPO 族 `kto_pair`（成对），非 legacy 非成对 desirable/undesirable+KL |
| GKD `lmbda`/`sft_alpha` | 恒 on-policy 生成；twinkle `GKDLoss` 为纯 JSD，未折入 SFT 项 |
| PPO critic | per-token value 走 DDP/AccelerateStrategy 路径，暂不覆盖 sequence-parallel / packed（critic 不开这些） |
| 偏好参考 | 仅 `mode='local'` in-process；ray/megatron 偏好参考未接 |
