# 训练迁移功能基线（legacy `swift pt` / `swift sft` 完整功能清单）

> 目的：把 legacy 的 **pt（预训练/CPT）** 与 **sft（指令微调）** 两条训练链路的功能面**逐项列全**，作为迁移到 dev 分层后的**功能回归基线**。dev 侧统一实现要保证：多模型路径（transformers / megatron / unsloth / sentence_transformers / plugin-model）**接口一致**，且下列每一项功能都可回归、行为准确。
>
> 与 `MODEL_MIGRATION.md`（模型）、`DATASET_MIGRATION.md`（数据集）、`ARGUMENTS_MIGRATION.md`（参数）、`PLUGIN_MIGRATION.md`（插件）配套：那几份记「字段/条目迁没迁」，本份记「pt/sft 这条**训练链路**整体有哪些功能」。
>
> 来源：`swift/cli/{pt,sft}.py`、`swift/pipelines/train/{sft,pretrain,tuner}.py`、`swift/pipelines/base.py`、`swift/trainers/{trainer_factory,arguments,mixin,seq2seq_trainer}.py`、`swift/arguments/*`、`swift/model/register.py`、`swift/tuner_plugin/*`、`swift/megatron/{pipelines,arguments,trainers}/*`、各 `*/mapping.py`。

---

## 0. 一句话结构

```
CLI (swift/cli/{pt,sft}.py)
  └─ Pipeline (SwiftSft / SwiftPretrain，继承 SwiftPipeline + TunerMixin)
       ├─ 参数 (SftArguments / PretrainArguments)
       ├─ 模型 (args.get_model_processor → transformers / unsloth / sentence_transformers / plugin)
       ├─ 模板 (args.get_template，set_mode('train'))
       ├─ 数据 (load_dataset / cached_dataset → encode → packing/lazy/streaming)
       ├─ tuner (TunerMixin.prepare_model → peft / unsloth / tuners_map)
       └─ Trainer (TrainerFactory.get_trainer_cls → Seq2SeqTrainer/Trainer/EmbeddingTrainer/RerankerTrainer)
```

Megatron 是一条**并行但独立**的栈：`MegatronSft(SwiftSft)` 复用 pipeline 的数据/模板逻辑，但模型走 meta-device + mcore-bridge，trainer 换成 `MegatronTrainer` 系列，参数换成 `MegatronSftArguments`。

---

## 1. 命令面与入口

### 1.1 CLI 入口
| 命令 | 文件 | 关键前置动作 |
|---|---|---|
| `swift pt` | `swift/cli/pt.py` | `try_use_single_device_mode()` → `pretrain_main()` |
| `swift sft` | `swift/cli/sft.py` | `try_use_single_device_mode()` → `try_init_unsloth()`（`--tuner_backend unsloth` 时提前 `import unsloth`）→ `try_init_ray()` → `sft_main()` |

- `try_use_single_device_mode`：单卡场景规避分布式初始化开销。
- `try_init_unsloth`：**必须在导入 transformers 前** import unsloth（unsloth 靠 import hook patch），故在 CLI 层用 `parse_known_args` 预读 `--tuner_backend`。
- `try_init_ray`：`use_ray` 时初始化 Ray runtime。

### 1.2 v5 路由（迁移目标侧）
- `USE_SWIFT_V5` 下 `pt`/`sft` 路由到 `swift.dev.cli.*`（`swift/dev/cli/pt.py`、`sft.py`）。
- v5 **无独立 megatron 子命令**，改用 `--backend megatron`（见 `swift/dev/cli/_megatron_compat.py`）。

### 1.3 Pipeline 基类（`swift/pipelines/base.py`）
`SwiftPipeline`（ABC）提供：`_parse_args`（`args_class` 解析）、`_set_seed`（`seed + rank`）、`main()` → `run()`。

---

## 2. pt 与 sft 的差异（极小，务必逐项对齐）

`SwiftPretrain(SwiftSft)` 仅换 `args_class = PretrainArguments`；`PretrainArguments(SftArguments)` 仅改两个默认值：

| 字段 | sft 默认 | pt 默认 | 含义 |
|---|---|---|---|
| `use_chat_template` | `True` | **`False`** | pt 用 generation/completion 模板而非 chat 模板 |
| `loss_scale` | `'default'` | **`'all'`** | pt 对**所有 token** 计损（含 system/user），sft 只对 response 计损 |

Megatron 侧同理：`MegatronPretrainArguments(MegatronSftArguments)` 同样只改这两项。

> 回归要点：pt 的 `truncation_strategy='split'` 只在预训练下可用（见 §6.6），且要求 `task_type=='causal_lm'` + `mode=='train'` + `not use_chat_template`。

---

## 3. 多模型路径（接口统一的五条路径）

统一入口：`BaseArguments.get_model_processor()`（`swift/arguments/base_args/base_args.py:331`）。它按参数分派到不同 loader，但**对外返回 `(model, processor)` 同构接口**。

| 路径 | 触发条件 | 实现 | 备注 |
|---|---|---|---|
| **transformers**（默认） | `tuner_backend != 'unsloth'` | `swift.model.get_model_processor`（`register.py:516`）→ `ModelLoader.get_model` | 主路径，覆盖 causal_lm/seq_cls/embedding/reranker/generative_reranker |
| **unsloth** | `--tuner_backend unsloth` | `load_by_unsloth(args)`（`register.py:45`）：多模态→`FastVisionModel`，MoE→`FastModel`，否则→`FastLanguageModel` | `get_peft_model` 在 tuner 阶段做（`tuner.py:205`）；仅支持 `lora`/`full`，不支持 longlora |
| **sentence_transformers** | `task_type=='embedding'` 且注册用 `SentenceTransformersLoader` | `SentenceTransformer(model_dir)`（`register.py:485`）+ `enable_input_require_grads` patch | embedding 专用加载器 |
| **megatron** | `swift megatron pt/sft`（或 v5 `--backend megatron`） | meta-device + `get_model_processor(load_model=False)`，权重经 mcore-bridge | 见 §10 |
| **plugin-model**（新 dev） | dev `ModelLoaderProtocol` 插件模型 | dev `swift/dev/model/loader/*` + twinkle `TransformersModel(model_loader=...)` | 迁移目标：保证与上述路径**接口相同** |

### 3.1 task_type 对模型加载头的影响（`register.py:277-331`）
- `seq_cls` / `reranker` → `AutoModelForSequenceClassification`，并 `tie_word_embeddings=False`；reranker 额外 patch `num_labels=1`。
- `embedding` → `patch_output_normalizer`（输出归一化）。
- `generative_reranker` → patch `lm_head`，用 yes/no token logits 打分（`get_generative_reranker_logits`）。
- `causal_lm` → `AutoModelForCausalLM`（默认）。
- reward model → `RewardModelLoader`（`register.py:508`）。

---

## 4. task_type 五值 → trainer / training_args / metric 映射

`TrainerFactory`（`swift/trainers/trainer_factory.py`）按 `rlhf_type`（若有）否则 `task_type` 分派。pt/sft 只涉及前五个（非 rlhf）：

| task_type | Trainer 类 | training_args 类 | 默认 eval_metric（`_init_metric`） |
|---|---|---|---|
| `causal_lm` | `Seq2SeqTrainer` | `Seq2SeqTrainingArguments` | `predict_with_generate` 时 `nlg`，否则 `loss` |
| `seq_cls` | `Trainer` | `TrainingArguments` | （用 `acc`） |
| `embedding` | `EmbeddingTrainer` | `TrainingArguments` | `loss_type=='infonce'`→`infonce`，否则`paired` |
| `reranker` | `RerankerTrainer` | `TrainingArguments` | `reranker` |
| `generative_reranker` | `RerankerTrainer` | `TrainingArguments` | `reranker` |

（rlhf 分派：`dpo/orpo/kto/cpo/rm/ppo/grpo/gkd` → `swift.rlhf_trainers.*`，不在 pt/sft 范围，仅列出以说明工厂完整性。）

- `get_training_args`：`asdict(args)` → 按 `training_args_cls` 的 `inspect.signature` **过滤字段** → `_prepare_training_args` 钩子 → 实例化。
- `metric_for_best_model`：`predict_with_generate`→`rouge-l`，否则`loss`；`greater_is_better` 由名字是否含 `loss` 推。
- seq_cls 在 `run()` 里补 `args.problem_type = args.problem_type or model.config.problem_type`。

---

## 5. 参数体系（完整旋钮面）

### 5.1 继承链（transformers 侧）
```
SftArguments(SwanlabArguments, TunerArguments, BaseArguments, Seq2SeqTrainingArguments)
BaseArguments(GenerationArguments, QuantizeArguments, DataArguments, TemplateArguments, ModelArguments, RayArguments)
Seq2SeqTrainingArguments(TrainArgumentsMixin, HfSeq2SeqTrainingArguments)   # causal_lm
TrainingArguments(TrainArgumentsMixin, HfTrainingArguments)                 # 其余 task_type
```

### 5.2 ModelArguments（`base_args/model_args.py`）
`model` `model_type` `model_revision` `task_type` `torch_dtype` `attn_impl`（sdpa/eager/flash_attn/flash_attention_2/3/4/flex_attention）`experts_impl`（grouped_mm/batched_mm/eager，需 transformers>=5）`new_special_tokens` `num_labels` `problem_type` `rope_scaling`（linear/dynamic/yarn 或 JSON）`device_map` `max_memory` `max_model_len` `local_repo_path` `init_strategy`。
派生：`_init_torch_dtype`→`_init_model_info`（拿 model_info/model_meta、task_type、num_labels、model_dir）；`_init_mixed_precision`（fp16/bf16 由 torch_dtype 推，MPS 全 False）；`_init_rope_scaling`（按 max_model_len 算 factor）；`_init_device_map`/`_init_max_memory`（mp+ddp 时按 local_rank 偏移）。

### 5.3 DataArguments（`base_args/data_args.py`）
`dataset`（`id_or_path:subset#count` 语法）`val_dataset` `cached_dataset` `cached_val_dataset` `split_dataset_ratio`（默认 0.）`data_seed`(42) `dataset_num_proc`(1) `load_from_cache_file`(False) `dataset_shuffle`(True) `val_dataset_shuffle`(False) `streaming`(False) `interleave_prob` `stopping_strategy`(first_exhausted/all_exhausted) `shuffle_buffer_size`(1000) `download_mode` `columns`(JSON 列映射) `strict`(False) `remove_unused_columns`(True) `disable_auto_column_mapping` `model_name` `model_author`（自认知占位符）`custom_dataset_info`。
派生：有 val_dataset 或 streaming 时 `split_dataset_ratio=0`；`_init_custom_dataset_info` 注册自定义 dataset_info.json；`_val_dataset_exists`。

### 5.4 TemplateArguments（`base_args/template_args.py`）
`template` `system`（串或 .txt）`max_length` `truncation_strategy`（delete/left/right/split，默认 delete）`max_pixels` `agent_template` `norm_bbox`(norm1000/none) `use_chat_template` `padding_side`(left/right，默认 right) `padding_free`(False) `loss_scale`(default) `sequence_parallel_size`(1) `is_binary_loss_scale` `template_backend`(swift/jinja，jinja 仅推理) `response_prefix` `enable_thinking` `preserve_thinking` `add_non_thinking_prefix`(True) `disable_ignore_empty_think`。
派生：`template_meta`/`template` 由 model_info+model_meta 解析；混合思考模型自动给 `loss_scale` 追加 `+ignore_empty_think`；`delete`→template 内部 `raise`。
- **loss_scale 策略**：基础 `default`/`last_round`/`all`；其他 `ignore_empty_think` + agent 系（`react`/`hermes`/`qwen`/`agentflan`/`alpha_umi`…）；可链式 `last_round+hermes+ignore_empty_think`。

### 5.5 QuantizeArguments（`base_args/quant_args.py`）
`quant_method`（bnb/hqq/eetq/quanto/fp8；awq/gptq/aqlm 为预量化自动识别）`quant_bits`(1/2/3/4/8/float8) `hqq_axis` `bnb_4bit_compute_dtype` `bnb_4bit_quant_type`(fp4/nf4) `bnb_4bit_use_double_quant`(True) `bnb_4bit_quant_storage`。
`get_quantization_config` 按方法构造 `BitsAndBytesConfig`/`HqqConfig`/`QuantoConfig`/`EetqConfig`/`FineGrainedFP8Config`；`get_modules_to_not_convert`（MoE gate、vision_tower、aligner、lm_head 不量化）→ QLoRA。

### 5.6 GenerationArguments（`base_args/generation_args.py`）
`max_new_tokens` `temperature` `top_k` `top_p` `repetition_penalty` `num_beams`(1) `stream` `stop_words` `logprobs` `top_logprobs` `structured_outputs_regex`（仅 vllm）。
`get_request_config`：仅 `task_type=='causal_lm'` 返回 `RequestConfig`（供 `predict_with_generate` / `_prepare_generation_config`）。

### 5.7 BaseArguments 自身（`base_args/base_args.py`）
`tuner_backend`(peft/unsloth) `tuner_type`(默认 lora) `adapters` `external_plugins` `custom_register_path`(3.x 兼容) `seed`(42) `model_kwargs`(JSON→env) `enable_npu_model_patch` `load_args`(训练默认 False) `load_data_args` **`packing`(False) `packing_length` `packing_num_proc`(1) `packing_strategy`(binpack/sequential) `lazy_tokenize`** `use_hf` `hub_token` `ddp_timeout` `ddp_backend` `ignore_args_error` `use_swift_lora`。
派生：`_init_adapters`（下载）；`_init_ckpt_dir`+`load_args_from_ckpt`（从 args.json 回填 force_load_keys/load_keys）；`_import_external_plugins`；`max_length` 缺省取 `model_info.max_model_len`；`packing_length` 缺省取 `max_length`；`_init_lazy_tokenize`（多模态非 streaming/packing/group_by_length 时默认 True；lazy 与 packing/streaming 互斥）；`is_adapter`=（tuner_type != full）；`adapters_can_be_merged`={lora,longlora,llamapro,adalora}。

### 5.8 TunerArguments（`arguments/tuner_args.py`）
- **full/freeze**：`freeze_parameters` `freeze_parameters_regex` `freeze_parameters_ratio`(0~1) `trainable_parameters` `trainable_parameters_regex`；多模态 `freeze_llm`(False) `freeze_vit`(True) `freeze_aligner`(True)。优先级：trainable_* > freeze_*。
- **通用**：`target_modules`(['all-linear']) `target_regex` `target_parameters`(MoE nn.Parameter，需 peft>=0.17) `modules_to_save`。
- **lora**：`lora_rank`(8) `lora_alpha`(32) `lora_dropout`(0.05) `lora_bias`(none/all) `lora_dtype` `lorap_lr_ratio` `use_rslora` `use_dora`。
- **lora_ga**：`lora_ga_batch_size/iters/max_length/direction/scale/stable_gamma`。
- **init_weights**：true/false/gaussian/pissa/pissa_niter_N/olora/loftq/lora-ga；bone: bat/true/false。
- **fourierft**：`fourier_n_frequency`(2000) `fourier_scaling`(300)。
- **boft**：`boft_block_size/block_num/n_butterfly_factor/dropout`。
- **vera**：`vera_rank`(256) `vera_projection_prng_key/dropout/d_initial`。
- **adapter**：`adapter_act`(gelu) `adapter_length`(128)。
- **adalora**：`adalora_target_r/init_r/tinit/tfinal/deltaT/beta1/beta2/orth_reg_weight`。
- **llamapro**：`llamapro_num_new_blocks`(4) `llamapro_num_groups`。
- **reft**：`reft_layer_key/layers/rank/intervention_type/args`。
- 派生：`init_weights` 字符串 true/false→bool；`_init_multimodal_full`（full + 多模态时把 freeze_llm/vit/aligner 翻成 freeze_parameters，aligner 不冻则进 trainable，generator 恒冻）；`target_regex`→`target_modules`。

### 5.9 TrainArgumentsMixin（`swift/trainers/arguments.py`）—— 训练旋钮
- **batch/精度**：`per_device_train_batch_size`(1) `per_device_eval_batch_size`(1) `gradient_accumulation_steps`（缺省 `ceil(16/bs/world_size)`）`gradient_checkpointing`(True) `vit_gradient_checkpointing` `gradient_checkpointing_kwargs` `safe_serialization`(True) `max_shard_size`(5GB)。
- **优化**：`weight_decay`(0.1) `adam_beta2`(0.95) `lr_scheduler_type`(cosine) `lr_scheduler_kwargs` `optimizer`(插件名) `use_liger_kernel`。
- **日志**：`logging_first_step`(True) `logging_steps`(5) `report_to`(['tensorboard'])。
- **dataloader**：`dataloader_num_workers`（Win=0 else 1）`dataloader_persistent_workers` `dataloader_prefetch_factor`(缺省 2) `train_dataloader_shuffle`(True) `group_by_length`。
- **loss 开关**：`router_aux_loss_coef`(0.) `enable_dft_loss` `enable_channel_loss`。
- **多模态/extra**：`check_model`(True) `acc_strategy`(token/seq) `max_epochs`（覆盖 num_train_epochs）`aligner_lr` `vit_lr`（任一非空→`optimizer='multimodal'`）`use_logits_to_keep` `ds3_gather_for_generation`(True) `resume_only_model`(False) `tuner_backend`。
- **插件**：`optimizer` `loss_type`（choices=loss_map）`eval_metric` `callbacks` `mrl_dims`（Matryoshka，JSON dim→weight）`early_stop_interval`。
- **train-eval loop（EvalScope）**：`eval_use_evalscope` `eval_dataset` `eval_dataset_args` `eval_limit` `eval_generation_config` `extra_eval_args`。
- **galore**：`use_galore` + `galore_target_modules/rank/update_proj_gap/scale/proj_type/optim_per_parameter/with_embedding/quantization/proj_quant/proj_bits/proj_group_size/cos_threshold/gamma_proj/queue_size`。
- **lisa**：`lisa_activated_layers`(>0→加 `lisa` callback) `lisa_step_interval`(20)。
- **flash ckpt**：`use_flash_ckpt`（DLRover，先写共享内存再异步落盘，不支持 safetensors）。
- 派生：`_init_callbacks`（lisa / adalora(tuner_type==adalora) / early_stop / activation_cpu_offload(fsdp_config)）；liger 与 device_map 互斥；`_init_liger`+`_patch_liger_kernel`（修 logits_to_keep，剔除 cu_seq_lens_* kwargs）。

### 5.10 SftArguments 自身 + 派生（`arguments/sft_args.py`）
字段：`add_version`(True) `create_checkpoint_symlink`(False) `output_dir` `learning_rate`（full=1e-5，else=1e-4）`eval_strategy`（缺省对齐 save_strategy；无 val→'no'）`fp16` `bf16` `max_new_tokens`(64) `temperature`(0.) `load_args`(False) `zero_hpz_partition_size`(ZeRO++) `deepspeed_autotp_size`(AutoTP) `fsdp`。
`__post_init__` 关键顺序：
1. `resume_from_checkpoint`→abspath；`resume_only_model` 时 full 改 `model`、否则改 `adapters`。
2. `BaseArguments.__post_init__` → `_init_override`（output_dir、metric、lr、eval_strategy）→ `TunerArguments.__post_init__`。
3. `_check_padding_free`：packing→强制 padding_free=True；padding_free/packing **要求 attn_impl ∈ flash_attn/flash_attention_2/3/4**，否则 raise。
4. `vit_gradient_checkpointing` 缺省 = `not freeze_vit`；`optimizer` 缺省按 lorap_lr_ratio→lorap / use_galore→galore。
5. 校验 dataset 或 cached_dataset 至少一个非空。
6. `_handle_pai_compat`（PAI 任务：logging_dir 用 PAI tensorboard，关 add_version）。
7. `_init_deepspeed` / `_init_fsdp` / `_init_device`。
8. `accelerator_config` 缺省 `{'dispatch_batches': False}`；无 eval_dataset→`eval_strategy='no'`。
9. `training_args = TrainerFactory.get_training_args(self)`；`remove_unused_columns=False`；`_add_version`（output_dir 加版本号、logging_dir=runs、run_name）。
10. `report_to` 含 swanlab→`_init_swanlab`。

### 5.11 SwanlabArguments（`arguments/sft_args.py`）
`swanlab_token/project(ms-swift)/workspace/exp_name/notification_method/webhook_url/secret/sender_email/receiver_email/smtp_server/smtp_port/email_language(zh)/mode(cloud)`。通知方式：dingtalk/lark/email/discord/wxwork/slack；email 需齐 sender+receiver+smtp_server+port。注册进 `INTEGRATION_TO_CALLBACK['swanlab']`。

---

## 6. 数据集 / 模板 / 编码特性

### 6.1 数据准备总流程（`SwiftSft._prepare_dataset` / `_encode_dataset` / `_post_process_datasets`）
1. **来源**：`cached_dataset`/`cached_val_dataset`（`get_cached_dataset`，与 streaming 互斥）或 `args.load_dataset()`（普通/流式）。两者可 concat。
2. **编码**（`_encode_dataset`）：非 lazy/streaming 时用 `AddLengthPreprocessor`（只补 `lengths` 字段，兼容 cached_dataset）或 `EncodePreprocessor`（`truncation_strategy=='split'`）；多模态 batch_size=100，否则 1000；编码时临时把 `template.model=None` 避免序列化模型。
3. **后处理**（`_post_process_datasets`）：
   - 非 streaming 且非 split → `LazyLLMDataset(dataset, template.encode, strict, random_state=data_seed)`（惰性编码）。
   - `packing` → `PackingDataset`（非流式）/ `IterablePackingDataset`（流式），参数：num_proc / packing_length / packing_num_proc / packing_strategy / strict / load_from_cache_file。
   - 仅 `streaming`（无 packing）→ `EncodePreprocessor` 即时编码。
4. **展示**（`_show_dataset`）：master 打印首样本 `template.print_inputs`；非 lazy/streaming 时统计 token 长度分布（`_stat_dataset`）。
5. `predict_with_generate` 时 **val_dataset 跳过编码**（保留原始 messages 供生成）；`_save_val_dataset` 把从训练集切出的 val 存 `val_dataset.jsonl`。
6. grpo/gkd（`pre_process=False`）延迟到训练阶段编码——pt/sft 恒 `pre_process=True`。

### 6.2 packing
- `packing_strategy`：`binpack`（best-fit-decreasing，会重排样本，默认）/ `sequential`（next-fit 保序贪心，单开包，配合 `packing_num_proc=1`）。
- `packing_length` 缺省 = `max_length`。
- packing 强制 `padding_free=True`（`_check_padding_free`）；与 `lazy_tokenize` 互斥。

### 6.3 padding_free
- batch 内展平避免 padding，序列间因果隔离；**需 flash attention**（flash_attn/flash_attention_2/3/4）+ transformers>=4.44。
- 模板支持性：`template.support_padding_free`，None 时按「非多模态」判定；不支持则 raise。
- 支持 CPT/SFT/DPO/GRPO/KTO/GKD。

### 6.4 streaming
- 边读边处理，**必须设 `--max_steps`**（长度未知）；预处理只在 rank0，大 world_size 下可能成瓶颈；与 lazy_tokenize/cached_dataset 互斥；Megatron 下强制 `dataloader_num_workers=1`。

### 6.5 lazy_tokenize
- 缺省：多模态且非 streaming/packing/group_by_length → True，否则 False；cached_dataset 时 False。

### 6.6 truncation_strategy
- `delete`（默认，超长丢弃，template 内部 `raise`）/ `left` / `right` / `split`。
- **`split` 仅预训练可用**：要求 `task_type=='causal_lm'` + `template.mode=='train'` + `not use_chat_template`，且 `not lazy_tokenize`；把长样本切成多条避免浪费 token；与 cached_dataset 不兼容。多模态用 left/right 会保留全部图像 token 可能 OOM。

### 6.7 sequence_parallel
- `sequence_parallel_size>1` 时 `sequence_parallel.prepare(size, model, tokenizer, padding_free=...)`（`_prepare_model_tokenizer`）；支持 CPT/SFT/DPO/GRPO；trainer 侧有 `get_sp_dataloader`。

### 6.8 模板编码与 collator 内部行为（`swift/template/base.py`，迁移必须逐项对齐）
- **`encode` 入口分派矩阵**（`task_type` × `mode`）：
  - `causal_lm`：`train`/`transformers`/`vllm`/`lmdeploy`/`sglang`→`_encode_truncated(chosen)`；`rlhf`→`_rlhf_encode`；`kto`→`_kto_encode`。
  - `seq_cls`：`rlhf`→`_rlhf_encode`（并 pop `chosen/rejected_labels`+`loss_scale`）；否则 `_seq_cls_encode(chosen)`。
  - `embedding`→`_embedding_encode`（anchor_/positive_/negative_* + labels）；`reranker`/`generative_reranker`→`_reranker_encode`（pos=1/neg=0，拼接 chosen）。
  - 入口带 `@torch.inference_mode()` + `@retry_decorator(3)`；收尾统一处理 `channel`/`lengths`/`_extra_kwargs`。
- **`_encode`**：`swift` 后端走 `_swift_encode`（拼 context_list + loss_scale），`jinja` 后端走 `_jinja_encode`（`apply_chat_template`）；encoder-decoder 拆 prompt/answer；首 token `labels[0]=-100`、`loss_scale[0]=0`；非训练态清空 `*labels`/`*loss_scale`。
- **`_encode_truncated`**：按 `truncation_strategy` 处理超长——`left`/`right` 走 `_truncate`（保护 placeholder/多模态 token）；`raise` 抛 `MaxLengthError`；`split` 切成多条 batched（首 token 置 -100/0）。非 causal_lm task_type 会 pop `labels`/`loss_scale`。
- **`compute_sft_loss`**：`outputs=model(**inputs)`；有 `num_items_in_batch` 时 `loss *= ((labels[:,1:]!=-100).sum() / num_items_in_batch)`（修 HF#34263 的跨卡 token 归一）。
- **`data_collator` 分派**：`causal_lm`→`_data_collator`/`_rlhf_data_collator`/`_kto_data_collator`；`seq_cls`→`_seq_cls_data_collator`；`embedding`→`_embedding_data_collator`；`reranker`→`_reranker_data_collator`（训练态按 `MAX_POSITIVE_SAMPLES`/`MAX_NEGATIVE_SAMPLES` 随机采样，固定 `RandomState(42)`）。
- **`_data_collator` 内核**：`padding_side`（训练态取 self.padding_side，否则 left）；`padding_free`→`packing_row` 展平（要求 position_ids）；megatron/SP→强制 right padding + 生成 position_ids；`gather_keys`=[labels/loss_scale/position_ids/token_type_ids/mm_token_type_ids]；`pad_values`=[-100/0./0/0/0]；megatron 非 padding_free 构 2D causal `attention_mask`；3D position_ids（mrope）走 `_pad_3d_position_ids`；末尾 `_data_collator_mm_data` 合多模态字段；megatron 额外回填 `seq_lens`（CP 定位末 token）。
- **`_handle_megatron_cp`**：CP>1 时把 input_ids/labels/loss_scale/length/mm_token_type_ids 补齐到 `cp_size*2` 的整数倍。
- **`packing_row`**：input_ids/labels/loss_scale/position_ids/token_type_ids 直接 concat；3D position_ids 与 mm_token_type_ids 用 `torch.cat(dim=-1)`；无 position_ids 时按各段 length 生成。
- **`register_post_encode_hook`/`remove_post_encode_hook`**：多模态训练把 `_post_encode`（input_ids→inputs_embeds）注册为 forward pre-hook；DeepSpeed ZeRO3 下 patch `deepspeed.initialize` 把 hook 移到末尾。

---

## 7. Tuner 体系（`swift/pipelines/train/tuner.py`）

### 7.1 `TunerMixin.prepare_model`（统一入口）
- transformers<4.45 时在此 `apply_liger(model_type)`（>=4.45 由 transformers 内建）。
- **is_adapter 分支**：非 unsloth 且非 tuners_map → `model.requires_grad_(False)`；有 `resume_from_checkpoint`/`adapters` → `tuner.from_pretrained(..., is_trainable=True)`（tuners_map 用插件 Tuner，否则 `Swift`）；否则 `tuners_map[type].prepare_model` 或 `prepare_adapter`。之后把可训练的 fp16 参数转 fp32（修 peft #1249 unscale 报错）。
- **full 分支**：`model.train()` + `requires_grad_(True)` + `freeze_parameters(ratio/list/regex)` + `activate_parameters(trainable_*)`。
- `use_galore`：缺省 target=all-linear(+embedding)。
- deepspeed zero3：`_patch_modules_to_save_zero3`（同步 `ds_grads_remaining`）。

### 7.2 `prepare_adapter` 支持的 tuner
`lora` / `longlora`（swift LoRAConfig / peft LoraConfig / unsloth get_peft_model / lora-ga）、`adalora`、`llamapro`、`adapter`、`vera`、`boft`、`fourierft`、`reft`、`bone`。
- `get_target_modules`：`all-linear`→多模态用 `get_multimodal_target_regex`（受 freeze_llm/vit/aligner 影响）否则 `find_all_linears`；`all-embedding`→`find_embedding`。
- `get_modules_to_save`：`all-embedding`/`all-norm` 展开；seq_cls(reward) 追加 `v_head`。
- peft task_type 映射：EMBEDDING→None，RERANKER→SEQ_CLS，GENERATIVE_RERANKER→CAUSAL_LM。
- longlora：仅 LLAMA + transformers>=4.39.3，`replace_llama_attn` + `group_size_ratio=0.25`。
- adalora：需 `calculate_max_steps` 传 total_step。

### 7.3 tuner_plugin（`swift/tuner_plugin/*`）
`Tuner` ABC（`prepare_model` / `save_pretrained` / `from_pretrained`）；`tuners_map = {ia3: IA3Tuner, lora_llm: LoRALLMTuner, dummy: DummyTuner}`。`tuner_type` 合法集 = 内建 {lora,full,longlora,adalora,llamapro,adapter,vera,boft,fourierft,reft,bone} ∪ tuners_map。

---

## 8. 分布式与并行（transformers 侧）

### 8.1 DeepSpeed（`_init_deepspeed`）
- 预设：`zero0/zero1/zero2/zero3/zero2_offload/zero3_offload`（映射到 `swift/config/*.json`），或直接传 JSON/路径。
- 与 `device_map`（mp）不兼容（非 ray 时 raise）。
- **ZeRO++**：`zero_hpz_partition_size`（节点内模型分片、节点间数据分片；grad_norm NaN 时建议 fp16）。
- **AutoTP**：`deepspeed_autotp_size`（需 deepspeed=zero0/1/2，仅全参；自动置 `gather_16bit_weights_on_model_save`）。
- **elastic**：callbacks 含 `deepspeed_elastic` → `prepare_deepspeed_elastic_config`；resume 需 universal ckpt（`get_resume_checkpoint_until_find_ucp`）。

### 8.2 FSDP2（`_init_fsdp`）
- `--fsdp fsdp2` → `swift/config/fsdp2.json`，设 `FSDP_VERSION` env、`TORCH_NCCL_AVOID_RECORD_STREAMS=1`。
- 与 device_map、DeepSpeed 互斥。
- 兼容性校验：`save_only_model=True` + `SHARDED_STATE_DICT` → raise；`gradient_checkpointing` 与 `fsdp_config.activation_checkpointing` 同开→自动关前者。

### 8.3 其它
`ddp_backend`(nccl/gloo/mpi/ccl/hccl/cncl/mccl) `ddp_timeout`(18000000)；`ds3_gather_for_generation`（zero3 生成时聚合参数）；Ray（`@RayHelper.worker(group=['default'])` / `@RayHelper.function(group='default')`）。

---

## 9. 插件映射表（迁移后需保持可注册/可分派）

| 类别 | 映射 | 键 |
|---|---|---|
| optimizer | `swift/optimizers/mapping.py:optimizers_map` | `default` `galore` `lorap` `muon` `muonclip` `multimodal` |
| loss | `swift/loss/mapping.py:loss_map` | `cross_entropy` `cosine_similarity` `contrastive` `online_contrastive` `infonce` `pointwise_reranker` `listwise_reranker` |
| eval_metric | `swift/metrics/mapping.py:eval_metrics_map` | `acc` `nlg` `infonce` `paired` `reranker` |
| callback | `swift/callbacks/mapping.py:callbacks_map` | `activation_cpu_offload` `adalora` `deepspeed_elastic` `early_stop` `graceful_exit` `lisa` `perf_log` |
| tuner | `swift/tuner_plugin/mapping.py:tuners_map` | `ia3` `lora_llm` `dummy` |
| megatron callback | `swift/megatron/callbacks/mapping.py` | `print` `default_flow` `swanlab` `wandb` `tensorboard` |

- `external_plugins`（旧 `custom_register_path`）：`import_external_file` 注册自定义 plugin.py。

### 9.1 插件契约（构造签名统一为 `(args, trainer)`，迁移时须保持）
- **optimizer**（`optimizers/base.py:OptimizerCallback`）：`create_optimizer_and_scheduler(num_training_steps)` → 分别设 `trainer.optimizer`/`trainer.scheduler`；`default` 委托 HF `Trainer.create_optimizer/create_scheduler`。变体：`galore`（GaLore 投影）、`lorap`（lorap_lr_ratio 分组）、`muon`/`muonclip`（Muon + clip）、`multimodal`（vit_lr/aligner_lr 分组）。
- **loss**（`loss/base.py:BaseLoss`）：`__call__(outputs, labels, *, num_items_in_batch, loss_scale, **kwargs)`。`cross_entropy`→`CustomCrossEntropyLoss`（走 `per_token_loss_func`，`sum/num_items_in_batch`，缺省 `num_items_in_batch=(labels[:,1:]!=-100).sum()`）；embedding 系 `cosine_similarity`/`contrastive`/`online_contrastive`/`infonce`；reranker 系 `pointwise_reranker`/`listwise_reranker`。
- **eval_metric**（`metrics/base.py:EvalMetrics`）：`compute_metrics(EvalPrediction)` + `preprocess_logits_for_metrics`。`acc`（token/seq 策略，seq 支持 cu_seqlens padding_free）、`nlg`（rouge/jieba）、`infonce`/`paired`（embedding）、`reranker`。
- **callback**（`callbacks/base.py:TrainerCallback(HfTrainerCallback)`）：`lisa`（on_step_begin 按 `lisa_step_interval` 随机激活 n 层、仅 full）、`adalora`（on_step_end 调秩）、`early_stop`（on_save 查 metric）、`perf_log`（MFU/FLOPS 日志）、`activation_cpu_offload`（激活值 CPU offload）、`deepspeed_elastic`（弹性标记）/`graceful_exit`（on_step_end/on_save 写优雅退出标记）。

---

## 10. Megatron 专属栈（`swift/megatron/*`）

### 10.1 Pipeline（`megatron/pipelines/train/sft.py`）
`MegatronSft(SwiftSft)`，`args_class=MegatronSftArguments`：
- `prepare_trainer`：embedding→`MegatronEmbeddingTrainer`，reranker/generative_reranker→`MegatronRerankerTrainer`，else→`MegatronTrainer`。
- `__init__`：**跳过 SwiftSft.__init__**，直接 `SwiftPipeline.__init__`；NPU 时 `apply_mindspeed_patches`（`attention_backend!='local'`→`use_flash_attn=True`）；`torch.device('meta')` 下 `get_model_processor`（多模态且 template.use_model→`return_dummy_model=True`，否则 `load_model=False`）；`_prepare_template`；`save_args`；`template.use_megatron=True`。
- `_set_seed`：pass（Megatron 自己管种子）。
- `run`：`_prepare_dataset` → `args.init_iters(train,val)` → `trainer.train` → finally `_handle_trainer_state` + `plot_images` + `logging.jsonl`；结尾 `dist.destroy_process_group()`（**不放 finally**，避免异常挂起）。

### 10.2 参数（`megatron/arguments/*`）
`MegatronSftArguments(MegatronBaseArguments)`；`MegatronBaseArguments(MegatronArguments, BaseArguments)`：`sequence_parallel_size=context_parallel_size`；`packing→padding_free=True`；`seq_length=packing_length or max_length`；streaming→`dataloader_num_workers=1`；`skip_megatron_init`（ray 跳过分布式 init）。
`MegatronPretrainArguments`：同 pt 只改 `use_chat_template=False` + `loss_scale='all'`。

**MegatronTunerMixin**：`tuner_type`(lora/full/lora_llm，默认 full) `freeze_llm/vit/aligner` `freeze_parameters(_regex/_ratio)` `trainable_parameters(_regex)` `target_modules/target_regex/modules_to_save` `lora_rank/alpha/dropout/bias/dtype/use_rslora`。（`freeze_parameters_ratio` 与 PP>1 互斥。）

**MegatronArguments 核心旋钮**：
- batch/iters：`micro_batch_size`(1) `global_batch_size`(16) `train_iters` `num_train_epochs`。
- recompute：`recompute_granularity`(selective/full/none) `recompute_method`(uniform/block) `recompute_num_layers` `recompute_modules`(['core_attn'])。
- fusion：`masked_softmax_fusion` `bias_dropout_fusion` `bias_activation_fusion` `apply_rope_fusion` `gradient_accumulation_fusion` `cross_entropy_loss_fusion` `cross_entropy_fusion_impl`(native/te)。
- optimizer：`optimizer`(adam/sgd/muon/dist_muon) `optimizer_cpu_offload` `optimizer_offload_fraction` `optimizer_cuda_graph` `use_precision_aware_optimizer` `main_grads_dtype` `main_params_dtype` `exp_avg_dtype` `exp_avg_sq_dtype`；muon 系列 `muon_momentum/split_qkv/use_nesterov/scale_mode/fp32_matmul_prec/coefficient_type/num_ns_steps/tp_mode/extra_scale_factor/scalar_optimizer`；`adam_beta1/beta2/eps` `sgd_momentum` `clip_grad`(1.)。
- lr：`lr` `min_lr` `lr_decay_style`(constant/linear/cosine/inverse-square-root/WSD) `lr_decay_iters` `lr_warmup_init/iters/fraction` `lr_wsd_decay_style/iters`；weight_decay 系 `weight_decay_incr_style/start_weight_decay/end_weight_decay`。
- 并行：`tensor_model_parallel_size` `pipeline_model_parallel_size`(+`decoder_first/last_pipeline_num_layers`、`account_for_embedding/loss_in_pipeline_split`、`pipeline_model_parallel_layout`) `virtual_pipeline_model_parallel_size`(+`microbatch_group_size_per_vp_stage`) `context_parallel_size`(+`cp_comm_type`、`cp_partition_mode` zigzag/contiguous) `expert_model_parallel_size` `expert_tensor_parallel_size` `sequence_parallel`；overlap 系 `overlap_p2p_comm/batch_p2p_comm/tp_comm_overlap/overlap_grad_reduce/overlap_param_gather(+_with_optimizer_step)/align_param_gather/align_grad_reduce/nccl_comm_warmup`。
- 分布式优化器：`use_distributed_optimizer`(True) `use_megatron_fsdp` `data_parallel_sharding_strategy`(no_shard/optim/optim_grads/optim_grads_params) `data_parallel_random_init`。
- data：`seed` `train_dataloader_shuffle` `dataloader_num_workers`(4)/`pin_memory`/`persistent_workers`/`prefetch_factor` `data_sharding` `group_by_length` `te_rng_tracker` `padding_free`(True) `mlp_padding_free`。
- ckpt：`save_steps`(500) `no_save_optim/rng` `mcore_model` `mcore_adapter` `no_load_optim/rng` `finetune`(True) `perform_initialization` `use_cpu_initialization` `async_save` `save_total_limit` `metric_for_best_model` `greater_is_better` `use_persistent_ckpt_worker` `dist_ckpt_save_pre_mcore_014` `dist_ckpt_optim_fully_reshardable`。
- 精度：`fp16/bf16` `attention_softmax_in_fp32` `accumulate_allreduce_grads_in_fp32` `apply_query_key_layer_scaling`；**fp8** `fp8_format/recipe/param_gather/amax_history_len/amax_compute_algo`；**fp4** `fp4_format/recipe/param_gather`。
- MoE：`moe_router_load_balancing_type` `moe_router_dtype` `moe_token_dispatcher_type`(allgather/alltoall/flex) `moe_enable_deepep` `moe_grouped_gemm` `moe_permute_fusion` `moe_aux_loss_coeff` `moe_z_loss_coeff` `moe_shared_expert_overlap` `moe_layer_recompute` `moe_expert_capacity_factor` `moe_pad_expert_input_to_capacity` `moe_token_drop_policy`。
- MTP：`mtp_num_layers` `mtp_loss_scaling_factor` `mtp_decoder_input_detach` `mtp_shared_weights`。
- attention/其它：`attention_backend`(flash/fused/unfused/local/auto) `calculate_per_token_loss` `manual_gc(_steps/_eval)` `bridge_backend`(mcore-bridge/megatron-bridge) `save_safetensors` `merge_lora` `max_shard_size`；visual `vit_gradient_checkpointing(_kwargs)/vit_attn_impl/vit_lr/aligner_lr`；dsa `dsa_indexer_loss_coeff/use_sparse_loss/apply_dsa_kernel_fusion`；deepseek-v4 `csa_dense_mode/use_fused_mhc/mhc_recompute_layer_num`；`megatron_extra_kwargs` `language_model_only` `check_model` `torch_dtype` `rope_scaling` `apply_wd_to_qk_layernorm` `linear_decoupled_in_proj` `enable_dft_loss` `enable_channel_loss` `task_type`(causal_lm/seq_cls/embedding/generative_reranker) `num_labels` `problem_type` `mrl_dims`。
- 日志/评估：`report_to` `logging_steps` `tensorboard_dir/queue_size` `wandb_project/exp_name` `swanlab_project/exp_name` `eval_iters`(-1) `eval_steps`。

> `MegatronArguments` 还含 `RLHFMegatronArgumentsMixin`（rlhf_type/beta/grpo/gkd/teacher/vllm/reward 系列），属 rlhf 范畴，pt/sft 不触发（`rlhf_type is None` 时 `__post_init__` 直接 return）。

### 10.3 Megatron trainer 家族实现（`megatron/trainers/*`，pt/sft 走 causal_lm/seq_cls/embedding/reranker）
- **`BaseMegatronTrainer.__init__`**：`prepare_model`（`get_mcore_model` → `bridge.load_weights` → `prepare_mcore_model` → `wrap_model`）→ `get_optimizer_and_scheduler`（adam/sgd/muon，`OptimizerConfig` 由 args 同名字段过滤构造）→ `_get_data_collator`（`template.data_collator` + `get_padding_to`）→ `TrainerState(max_steps=train_iters)`→（new_special_tokens/seq_cls 时 `_initialize_embedding`）→ `_load_checkpoint`（mcore_model/mcore_adapter）→ callbacks（`megatron_callbacks_map`）。
- **训练循环**：`train`→`setup_training`（`setup_model_training` 配 grad_scale_func/no_sync_func/param_sync_func/finalize_model_grads + NCCL warmup + VPP 多 data_iterator）→ `while iteration<train_iters: run_train_step`。`run_train_step`：`train_step`（`get_forward_backward_func` 跑 `num_microbatches`）→ optimizer.step → `logical_and_across_model_parallel_group(update_successful)` → `reduce_max_stat_across_model_parallel_group(grad_norm)` → scheduler.step；按 `state.should_log/should_eval/should_save` 触发 on_log/evaluate/save_checkpoint。
- **`forward_step`（MegatronTrainer）**：`get_batch`→`prepare_batch`（PP 切片 `get_batch_on_this_pp_rank` + `packed_seq_params`（padding_free）+ CP 切片 `get_batch_on_this_cp_rank`；NPU 生成 attention_mask）；seq_cls→`seq_cls_loss_func`（`get_last_tokens` 取末位 + MSE/CE/BCE 按 problem_type + acc），causal_lm→`loss_func`（`loss_mask=labels!=-100`，`enable_dft_loss` 乘 `exp(-loss)`，loss_scale 加权，`(sum, count)` 双元组 all_reduce，`enable_channel_loss`→`_compute_channel_loss` 按 channel 分组）。
- **`get_last_tokens`**：CP>1 先 `reconstruct_tensor_cp`；padding_free 用 `cu_seqlens_q+seq_lens-1` 定位，否则 `get_last_valid_indices(attention_mask)`。
- **embedding/reranker trainer**：`loss_func` 从 `loss_map[args.loss_type]` + `eval_metrics_map`（embedding→infonce/paired，reranker→reranker）装配；embedding 支持 MRL（`mrl_dims` 各截断维 `F.normalize` 加权）；reranker `prepare_model` 额外把 `tokenizer` 挂到 `language_model`（generative_reranker 打分用）。
- **`init_iters`**（`megatron_args.py`）：`save_strategy=='epoch'`→按 `len(dataset)//step_batch_size` 推 `save_steps`/`eval_steps`；`num_train_epochs` 非空且数据集有 `__len__`→`train_iters = dataset_sample*num_train_epochs//global_batch_size`（streaming 必须显式传 `--train_iters`）；`eval_iters<0`→按 val 集推（streaming 报错）；val 集不足一个 step 时置 `eval_iters=0`。
- **`save_checkpoint`**：`save_mcore_checkpoint`（dist ckpt，`peft_format=tuner_type=='lora'`）+（`save_safetensors`）`bridge.save_weights` 回写 HF 格式（processor 同存）；`merge_lora` 时额外存 `-merged` 目录；`_rotate_checkpoints`（`save_total_limit`，保护 best）。
- **dataloader**：非 streaming→`MegatronPretrainingRandomSampler`（train，带 consumed_samples/data_sharding/group_by_length）+ `MegatronPretrainingSampler`（val）；streaming→`build_streaming_dataloader`（`MegatronDataLoaderDispatcher`，group=DP）。
- **ckpt 转换**：HF↔mcore 权重双向转换由 `mcore-bridge`（`self.config.bridge`）承担——加载 `bridge.load_weights(models, model_dir[, peft_format, adapter_name])`，保存 `bridge.save_weights(...)`。

---

## 11. 训练循环特性（`swift/trainers/mixin.py`、`seq2seq_trainer.py`、`patcher.py`）

### 11.1 SwiftSft.run / train
- `run()`：数据准备 → seq_cls 补 problem_type → `save_args()` → `prepare_model`（TunerMixin）→ 记 `model_parameter_info` → `TrainerFactory.get_trainer_cls` → 构造 trainer（model/training_args/template/train_dataset/eval_dataset）→ `train()`。
- `train()`：`_get_resume_checkpoint` → `trainer.train(resume_checkpoint)` → finally `_save_trainer_state`（+ flash_ckpt `wait_latest_checkpoint`）。
- **resume 解析**（`_get_resume_checkpoint`）：`resume_from_checkpoint` 优先；否则 `use_flash_ckpt`→`trainer.get_resume_checkpoint()`；`deepspeed_elastic` 且缺 `latest_universal`→`get_resume_checkpoint_until_find_ucp()`。
- **state 保存**（`_save_trainer_state`）：`create_checkpoint_symlink`→建 best/last 软链；tensorboard→`plot_images`；`push_to_hub`；写 `logging.jsonl`（last/best ckpt、best_metric、global_step、log_history、memory）。

### 11.2 SwiftMixin.__init__ 装配顺序（`mixin.py`）
1. `IterableDataset` + `dataloader_num_workers>1` → 强制置 1（避免多 worker 重复流式样本）。
2. `optimizer_callback = optimizers_map[args.optimizer or 'default'](args, self)`——优化器插件在此实例化。
3. `check_model` → `check_local_model_is_latest`（`config_info` 携带 `seq2seq_mode` = pt/sft、trainer_class、trainer_backend）。
4. `custom_metrics = {'train'/'eval': defaultdict(MeanMetric)}`；`create_loss_and_eval_metric`（按 `loss_type`/`eval_metric` 从 loss_map/eval_metrics_map 装配）；`_get_callbacks`（合并 callbacks_map 插件 + HF 默认）。
5. transformers 5.x 下修 `gradient_state.num_steps=1`；`_fix_gradient_checkpointing` + `_patch_tasks`。
6. `resume_only_model` + `ignore_data_skip` → `resume_from_checkpoint=None`（只加载权重不恢复数据流）。

### 11.3 `_patch_tasks`（task_type 前向包装，迁移必须逐项复现）
- `SentenceTransformer`（embedding）→ patch `forward_transformer` / `forward_sentence_transformer`（内含 `revert_padding_free`）。
- `task_type ∈ {seq_cls, reranker, generative_reranker, embedding}` → 注册 `sp_gather_hook`（`gather_sequence_parallel_outputs`）+ `revert_padding_free_hook`；其中 `seq_cls`/`reranker` 再用 `transformers_seq_cls_forward` 包装 `forward`。
- `padding_free` 时把 `padding_side='left'`。

### 11.4 保存链路
- `_save_model`：委托 `template.save_callback`；分支 `SwiftModel`/`PreTrainedModel`/`PeftModel`/`SentenceTransformer`；flash_ckpt 走 `ckpt_agent.save`；`SentenceTransformer` 特殊 `save_context`（strip 前缀 `0.auto_model.`）+ 拷 `*.py`/`*.json`。
- `_save`：`_save_model` + `training_args.bin` + `_save_converted_model` + 拷 `args.json` + 移 `predict.jsonl` +（非 adapter）`save_checkpoint(processor)` + `origin_generation_config`。
- `_save_initial_model`/`_save_converted_model`：pissa/olora/lora-ga 需保存 `initial_model` 与 `converted`/`default` 权重；保护 `requires_grad`（规避 peft>=0.18.1 bug）。

### 11.5 flash-ckpt（DLRover）生命周期
`_get_last_checkpoint_step`（读 `dlrover_latest.txt`）/ `get_resume_checkpoint`（`train_state.max_steps==step` 时返回 None）/ `get_resume_checkpoint_until_find_ucp`（读 `ucp.txt`，elastic 用）/ `_save_flash_checkpoint`（`HfDeepSpeed`/`HfDdpCheckpointer`，**仅支持 DS/DDP，不支持 FSDP**）/ `_rotate_flash_checkpoints`。

### 11.6 训练期数值/显存修补
- `_fix_grad_norm_nan`：patch `Accelerator.clip_grad_norm_`，NaN 时清空 `p.grad`。
- `_fix_zero3_gather_all_parameters`：`SwiftModel`/`PeftModel` 时 `exclude_frozen_parameters=True`。
- `_prepare_gradient_checkpointing`：`use_cache=False` + `dynamic_gradient_checkpointing`（自动探测 ≥10 层的非 MoE `ModuleList`）+ vision_tower gc；末尾把 `args.gradient_checkpointing=False`（避免被 transformers 二次覆盖）。
- `get_use_logits_to_keep`/`prepare_logits_to_keep`（SP 下 `NotImplementedError`）/`get_cu_seqlens`（padding_free/packing 变长边界）。
- `train()`：多模态时 `register_post_encode_hook`；套 4 个 contextmanager（`hub.patch_hub` / `_fix_grad_norm_nan` / `_patch_skip_first_batches` / `_patch_deepspeed_load_checkpoint`）。

### 11.7 `_compute_acc`（按 task_type 分支）
- embedding → 跳过；seq_cls → regression 跳过 / multi_label `sigmoid>0.5` / 否则 `argmax`；causal_lm → `argmax` + SP gather + `compute_acc(acc_strategy, cu_seqlens)`；reranker → listwise 或 `>0`。
- `_evalscope_eval`（`@no_grad`）：`TaskConfig` + `run_task(EvalModel)`，**强制禁用 packing/padding_free**。

### 11.8 DataLoaderMixin
- `get_train_dataloader`：SP → `get_sp_dataloader`（`SequenceParallelSampler`）；否则 `BatchSamplerShard`（`drop_last`、`shuffle=train_dataloader_shuffle`、`data_seed`、`tp_size=deepspeed autotp_size`、`group_by_length`+lengths）+ `DataLoaderShard`（`seed_worker`）；`IterableDataset` → `DataLoaderDispatcher`（`prefetch*world_size`）。

### 11.9 Seq2SeqTrainer（causal_lm，`seq2seq_trainer.py`）
`Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer)`。
- `__init__`：`model_accepts_loss_kwargs=True`（修 4.46.2，会被 template 覆盖）；`predict_with_generate`→建 `TransformersEngine` + `jsonl_writer`（写 `predict.jsonl`）。
- `_patch_predict_with_generate`：换 `_predict_data_collator`，禁 `template.packing`/`padding_free`。
- `prediction_step`：生成模式 `unwrap_model_for_generation`（`gather_deepspeed3_params`）+ `generate_context` + `infer_engine.infer`，写 `predict.jsonl`，`Serializer.to_tensor` + `pad_sequence`。
- `_prepare_inputs`：SP prepare；`logits_to_keep`（unsloth 转 `int(sum)`）；MoE `router_aux_loss_coef`→config + `output_router_logits`；注入 `compute_loss_func`。
- `compute_loss`：pop `compute_loss_func`/`loss_scale`/`text_position_ids`/`channel`；条件 pop `labels`；调 `template.compute_sft_loss`；`per_token_loss_func(_sp)`；loss_scale `roll(-1)`；channel loss（`cu_seqlens`）；`num_items_in_batch`（SP 下 all_reduce）；MoE `aux_loss`；`average_tokens_across_devices`；`_compute_acc`（padding_free+seq→cu_seqlens）。
- `training_step`：套 `template.forward_context`。

### 11.10 其余 trainer
- `Trainer`（seq_cls 基类，`trainer.py`）：`_prepare_inputs`（SP 时 per-sample `labels` dim==1 临时 pop）；`_patch_loss_function`（labels→logits.device 修 device_map）；`compute_loss`（+`_compute_acc`，`num_items_in_batch`→`/grad_accum`）。
- `EmbeddingTrainer`（`embedding_trainer.py`）：`gather_function=gather_for_unpadded_tensors`；MRL（Matryoshka）按 `mrl_dims` 各截断维度 `F.normalize` 加权求和。
- `RerankerTrainer`（`reranker_trainer.py`）：`generative_reranker`→`get_last_valid_indices` 取末位 logits；`compute_loss_func`；`_compute_acc`。

### 11.11 patcher.py（monkey-patch transformers trainer，**易漏，务必迁移**）
替换 transformers 的 `DEFAULT_PROGRESS_CALLBACK`/`DEFAULT_CALLBACKS`/`PrinterCallback`：
- `add_train_message`：日志追加 `global_step/max_steps`、`elapsed/remaining_time`、`memory(GiB)`、`train_speed(s/it)`，并写 `logging.jsonl`。
- `DefaultFlowCallbackNew`：**末步强制** `should_evaluate`+`should_save`；`max_epochs` 早停。

### 11.12 predict_with_generate
causal_lm 评估时生成文本，配 `eval_metric='nlg'`（需 jieba）、`max_new_tokens`(64)、`temperature`(0.)。

---

## 12. 迁移回归检查清单（逐项须可复现）

- [ ] `swift pt` 与 `swift sft` 默认值差异（use_chat_template / loss_scale）
- [ ] 五 task_type → 正确 trainer / training_args / 模型头 / eval_metric
- [ ] 五模型路径（transformers / unsloth / sentence_transformers / megatron / plugin-model）接口同构
- [ ] tuner 全集（lora/longlora/adalora/llamapro/adapter/vera/boft/fourierft/reft/bone/full + ia3/lora_llm/dummy）
- [ ] freeze/trainable 优先级 + 多模态 freeze_llm/vit/aligner
- [ ] 数据：packing(binpack/sequential) / padding_free / streaming / lazy_tokenize / cached_dataset / split_dataset_ratio / truncation(delete/left/right/split)
- [ ] sequence_parallel（prepare + sp_dataloader）
- [ ] 量化 QLoRA（bnb/hqq/eetq/quanto/fp8 + modules_to_not_convert）
- [ ] DeepSpeed（zero0-3/offload/ZeRO++/AutoTP/elastic）+ FSDP2 + 互斥校验
- [ ] 插件映射（optimizer/loss/metric/callback/tuner）可注册可分派
- [ ] resume（普通 / flash_ckpt / deepspeed_elastic universal ckpt / resume_only_model）
- [ ] predict_with_generate + nlg metric
- [ ] use_logits_to_keep / gradient_checkpointing / liger
- [ ] galore / lisa / lorap / multimodal optimizer 派生
- [ ] swanlab / wandb / tensorboard report + plot_images + logging.jsonl + push_to_hub
- [ ] Megatron 并行全旋钮（TP/PP/CP/EP/VPP/sequence_parallel + overlap + recompute + fp8/fp4 + MoE + MTP + muon）
- [ ] add_version / create_checkpoint_symlink / save_args / load_args_from_ckpt
- [ ] `_patch_tasks`：SP gather hook + padding_free revert hook + seq_cls/reranker 前向包装 + SentenceTransformer 前向 patch
- [ ] `patcher.py` 三个 monkey-patch（train_speed/memory/time 日志 + 末步强制保存/评估 + max_epochs 早停）
- [ ] `template.encode` 全分派矩阵 + `compute_sft_loss` 的 `num_items_in_batch` 缩放（HF#34263）
- [ ] `_data_collator`：padding_free packing_row / megatron CP 补齐 / 3D position_ids / 2D causal attention_mask
- [ ] flash-ckpt 生命周期（dlrover_latest.txt / ucp.txt / replace_index_file；**仅 DS/DDP，不支持 FSDP**）
- [ ] pissa/olora/lora-ga 的 initial_model + converted 保存
- [ ] _fix_grad_norm_nan / _fix_zero3_gather_all_parameters / dynamic_gradient_checkpointing
- [ ] MRL（mrl_dims）/ generative_reranker 末位 logits / DeepSpeed elastic compute_elastic_config
- [ ] Megatron trainer 家族：forward_step / loss_func（dft/channel）/ get_last_tokens / prepare_batch(PP+CP) / init_iters / mcore-bridge ckpt 双向转换
