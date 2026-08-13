# swift/ Patch 清单与 swift.dev 需求分析

> 范围：`swift/` 目录下的 monkey-patch（不含嵌套的 `twinkle/` 仓库）。
> 每条给出：**功能** + **触发场景** + **swift.dev 是否仍需要**。
>
> **判定依据**：新的 `swift.dev` 把「建模型」委托给 **twinkle + mcore-bridge**，有自己的 recipes / train_loop / processor / builders。因此可粗分三类结论：
> - ✅ **仍需要**：能力与后端无关，dev 尚未覆盖，迁移时需在 dev 侧重建或复用。
> - ⚠️ **部分/条件需要**：dev 目前未做，但一旦支持对应特性（量化 / RLHF / 特定模型 / NPU）就需要。
> - ❌ **不再需要**：twinkle / mcore-bridge 已接管，或属于被 dev 架构淘汰的 legacy 加载路径。
>
> 结论为**基于架构的分析判断**，非逐条运行验证；迁移某域前应再核对。

---

## 1. 通用模型加载 patcher — `model/patcher.py`

| Patch | 功能 / 触发场景 | dev 是否需要 |
|---|---|---|
| `patch_fixed_float_dtype` | forward hook 把输出转成固定 float dtype；混合精度下部分模块输出 dtype 不一致时用。 | ⚠️ twinkle 的 mixed_precision 已管 dtype，仅特殊模型可能需要 | 没有调用的地方
| `patch_fixed_device` | forward hook 把输出搬到指定 device；device_map 多卡场景。 | ❌ dev 走 twinkle strategy，不用 HF device_map | 已经不需要了
| `patch_output_clone` | forward hook clone 输出避免 inplace 问题。 | ⚠️ 个别模型 backward inplace 报错时才需要 | 已经加了 patch，在多模态+gradient_checkpointing 的时候需要
| `patch_get_input_embeddings` | 重定向 `get_input_embeddings` 到指定 key；模型未标准暴露 embedding 时。 | ⚠️ 视模型而定 | reentrant=True 的时候需要，目前应该不需要了
| `patch_output_normalizer` | 把 lm_head 换 identity + hook 做末 token 池化 + L2 normalize（embedding 任务）。 | ❌ **twinkle `MegatronEmbeddingPatch` / `_pool_last_valid_token` 已接管** |
| `patch_output_to_input_device` | hook 把输出搬回输入所在 device；pipeline/多卡。 | ❌ 同 device_map，dev 不用 | 不需要了
| `patch_device_map` | 上下文内改 `_get_no_split_modules`，让 HF device_map 切分正常。 | ❌ dev 不用 HF device_map | 不需要了
| `patch_ignore_check_imports` | 上下文内跳过 HF 动态模块 import 检查；加载 remote code 模型。 | ⚠️ 加载 trust_remote_code 模型仍可能有用 | 已经迁移 twinkle
| `_patch_sequence_classification` | 把 lm_head 换 identity + 新建 `score` Linear(num_labels) + 包 forward 按 problem_type 出 loss。 | ❌ **dev 已用 `_apply_seq_cls_head`（AutoModelForSequenceClassification）+ twinkle `SeqClsLoss` 重建** |
| `patch_automodel_for_sequence_classification` | 上下文内 patch `from_pretrained`，加载后自动挂 seq_cls 头 + 补缺失 `__init__`。 | ❌ 同上，dev 走 config-route 直接建头 |
| `patch_automodel` | 上下文内 patch `from_pretrained` 走 swift 自定义加载（dummy model / 各种兼容）。 | ❌ **twinkle `TransformersModel` 接管加载** |
| `patch_mp_ddp` | 配置 model-parallel + DDP 组合。 | ❌ dev 走 twinkle strategy（accelerate/deepspeed/fsdp/megatron） |
| `patch_get_dynamic_module` | 修 HF 动态模块获取。 | ⚠️ remote code 场景 |  已迁移到 twinkle
| `patch_tp_plan` | 修 `_tp_plan`（HF tensor-parallel 计划）。 | ❌ dev 的 TP 由 mcore-bridge/twinkle 管 | 不需要了
| `patch_attach_align_device_hook_on_blocks` | 修 accelerate 的 device hook 挂载。 | ❌ device_map 相关 | 用于 convert，已经迁移
| `patch_module_forward` | 通用替换某 module 的 forward 的工具。 | ⚠️ 作为工具函数，dev 若要自定义 forward 可复用 | device_map+generative_reranker 使用，废弃

---

## 2. 模型注册 / 量化兼容 — `model/register.py`

| Patch | 功能 / 触发场景 | dev 是否需要 |
|---|---|---|
| `_patch_generative_reranker` | patch lm_head 出 `yes-no` 差做 generative reranker 分数。 | ❌ **dev 已用 twinkle `TransformersGenerativeRerankerPatch` + bridge generative 头重建** |
| `_patch_distributed_function` | 加载期临时替换 `dist` 相关函数（避免误初始化进程组）。 | ⚠️ dev 加载在 twinkle 内，若无同等保护可能需要 | unsloth 需要，应进入 recipe
| `_patch_awq_compat` | AWQ 量化模型加载兼容。 | ⚠️ dev 支持 AWQ 加载训练时需要 | transformers 新版本不需要

---

## 3. 各模型专属 patch — `model/models/`

> 共性：都是"某模型在某 transformers 版本下的 bug/兼容/多卡/dtype"补丁，**与训练后端无关**。dev 一旦支持对应模型，多数**仍需要**（应迁移到 dev 的 model builder 或保留共享）。

| Patch | 功能 / 触发场景 | dev 是否需要 |
|---|---|---|
| `baichuan.patch_baichuan2_lm_head_forward` | Baichuan2 lm_head 归一化权重。 | ⚠️ 用 Baichuan2 时需要 | 不迁移
| `deepseek._apply_multi_gpu_patch` | DeepSeek 多卡 forward 兼容。 | ⚠️ 用 DeepSeek 多卡时需要 | device_map，不迁移
| `gemma._patch_gemma4_forward` | Gemma4 forward 兼容。 | ⚠️ 用 Gemma4 时需要 |迁移为通用 patch
| `glm._patch_tokenizer` | GLM tokenizer 修正。 | ⚠️ 用 GLM 时需要 | 老 glm2、3 需要，不迁移
| `internlm.patched_enable_input_require_grads` | InternLM 梯度输入开关兼容。 | ⚠️ 用 InternLM 时需要 | use_reentrant=False时不需要
| `llava._patch_llava` | LLaVA 多模态加载兼容。 | ⚠️ 用 LLaVA 时需要 | 模型比较老，不需要了
| `minicpm._patch_minicpmv_device_map` | MiniCPM-V device_map 修正。 | ⚠️/❌ device_map 相关，dev 视情况 | device_map 的不需要了
| `qwen.patch_qwen_vl_utils` | Qwen-VL 视觉处理工具兼容。 | ⚠️ 用 Qwen-VL 时需要 | 不需要了
| `qwen.patch_Qwen3VLMoeTextExperts_dtype` | Qwen3-VL-MoE experts dtype 修正。 | ⚠️ 用该模型时需要 | 旧版本 bug，不需要
| `qwen._patch_deepstack_process` | Qwen deepstack 处理。 | ⚠️ 用该特性时需要 |已迁移
| `qwen._patch_qwen3_5_linear_attention_sequence_parallel` | Qwen3.5 线性注意力 + SP 兼容。 | ⚠️ 用该模型 + SP 时需要 | 已迁移
| `qwen._patch_qwen3_tts_forward` | Qwen3-TTS forward 兼容。 | ⚠️ 用 TTS 时需要 | patch 已迁移，模型拉起需要额外处理
| `stepfun._patch_step_audio2_mini` | StepFun 音频模型兼容。 | ⚠️ 用该模型时需要 | twinkle 有自己的 loss，不需要
| `model/utils.py:_patch_conv3d` | Conv3D 兼容（部分多模态）。 | ⚠️ 相关多模态时需要 |
已迁移
---

## 4. 推理引擎 — `infer_engine/`（vLLM / lmdeploy）

> 全部是 **推理后端（vLLM/lmdeploy）兼容/规避 bug**。dev 目前聚焦训练；**推理/rollout 走对应引擎时才需要**。GRPO 等 RLHF rollout 依赖这些。

| Patch | 功能 / 触发场景 | dev 是否需要 |
|---|---|---|
| `patch.py:patch_auto_config` / `patch_auto_tokenizer` | 推理侧 AutoConfig/Tokenizer 加载兼容。 | ⚠️ 推理/部署时 | 暂时不迁移，vllm 使用默认 config
| `utils.py:patch_lmdeploy` | lmdeploy 引擎兼容。 | ⚠️ 用 lmdeploy 时 | 不迁移
| `utils.py:patch_npu_vllm` | NPU 上 vLLM 兼容。 | ⚠️ NPU + vLLM | 存疑，暂时不动
| `utils.py:patch_vllm_triton_device_guard` | vLLM triton device guard 修正。 | ⚠️ vLLM | colocate 需要，暂时不迁移
| `utils.py:patch_vllm_memory_leak` | 规避 vLLM 显存泄漏。 | ⚠️ vLLM |vllm 0.7.3，暂时不迁移
| `utils.py:patch_vllm_abort_seq_group` | vLLM abort 序列组兼容。 | ⚠️ vLLM | 同上
| `utils.py:patch_vllm_engine` | vLLM engine 综合兼容。 | ⚠️ vLLM | 同上
| `vllm_engine.py:_patch_vllm_dp_coordinator_timeout` | vLLM DP coordinator 超时。 | ⚠️ vLLM DP | twinkle 不需要 dp，因此暂时不迁移
| `vllm_engine.py:_patch_rope_validation_ignore_keys` | 跳过 rope 校验 key。 | ⚠️ vLLM | 0.18 需要，可以不迁移
| `vllm_engine.py:_patch_auto_config` | vLLM 侧 config 兼容。 | ⚠️ vLLM | 是否可以转本地文件存储，或其他方式，patch 暂时不迁移
| `vllm_engine.py:patch_remove_log` | 降噪 vLLM 日志。 | ⚠️ vLLM | 没必要迁移
| `lmdeploy_engine.py:_patch_pipeline` | lmdeploy pipeline 兼容。 | ⚠️ lmdeploy | lmdeploy 不管

---

## 5. Megatron — `swift/megatron/`

> 注意：**这些是 legacy 的 Megatron 后端补丁**。dev 的 Megatron 走 **twinkle + mcore-bridge**，其中一部分（如 mcore-bridge 挂载）twinkle 已自己做。但涉及 **Megatron-core / torch dist checkpoint / 超时** 的底层规避，dev 侧可能仍需等价处理。

| Patch | 功能 / 触发场景 | dev 是否需要 |
|---|---|---|
| `init.py:_patch_mcore_bridge` | 给 mcore-bridge `GPTBridge` 打补丁。 | ❌ **dev 直接用 mcore-bridge，twinkle 负责集成** | 合并到 megatronmodel 中
| `init.py:_patch__batched_p2p_ops` | Megatron PP 的 P2P 通信修正。 | ⚠️ dev 若遇同问题需等价处理（twinkle 可能已处理） | 已迁移
| `init.py:_patch_torch_FileSystemReader` | dist checkpoint 读取分片进度条/兼容。 | ⚠️ 分布式 ckpt 场景 | 已迁移
| `init.py:_patch_validate_non_overlapping_shards_metadata` | 跳过分片元数据重叠校验。 | ⚠️ dist ckpt 场景 | 已迁移
| `init.py:_patch_unified_memory` | 统一内存（UVM）支持。 | ⚠️ 特定硬件 | 老版本需要
| `trainers/base.py:_patch_get_param_groups` | legacy Megatron trainer 参数分组。 | ❌ dev 有自己的 optimizer/param-group 逻辑 | 已添加替代方案
| `utils/patcher.py:patch_torch_dist_shard` | torch dist checkpoint 分片线程数。 | ⚠️ **dev 的 convert 已在用**（见 `dev/recipes/convert.py`），需要 | 已迁移
| `utils/patcher.py:patch_merge_fn` | state_dict 合并函数。 | ⚠️ ckpt 转换/合并时 | 老版本代码优化 不需要
| `utils/convert_utils.py:_patch_attention_fp32` | 转换精度校验时强制 attention fp32。 | ⚠️ convert precision 测试时 | convert 精度测试使用，存疑
| `utils/megatron_lm_utils.py:_patch_megatron_timeout` | 调 Megatron 分布式超时。 | ⚠️ 大规模训练防超时 | timeout 设置，不需要迁移
| `utils/router_replay_utils.py:apply_router_replay_patch` | MoE router replay（复现路由）。 | ⚠️ MoE + router replay 时 | 需要把 R3 从 swift 迁移到 megatron 才能支持

---

## 6. 训练器 — `trainers/` & `rlhf_trainers/`

> legacy Trainer 基于 HF Trainer；dev 有自己的 `train_loop.py`（SFTLoop）。SFT 类补丁 dev **已用自身实现替代**；RLHF/vLLM-rollout 类补丁在 dev 支持 GRPO 前**未覆盖**。

| Patch | 功能 / 触发场景 | dev 是否需要 |
|---|---|---|
| `trainers/trainer.py:_patch_loss_function` | 用 HF 模型自带 `loss_function` 驱动 seq_cls loss。 | ❌ **dev 用 twinkle `SeqClsLoss` 在 loss 层分派** | 不需要
| `trainers/mixin.py:_patch_deepspeed_load_checkpoint` | DeepSpeed ckpt 加载兼容。 | ⚠️ dev + DeepSpeed resume 时 | 不需要迁移，对 trl 的 patch
| `trainers/mixin.py:_patch_tasks` | 按 task 调整 HF Trainer 行为。 | ❌ dev 有 task 分派机制 | 仅 sentence_transformers 需要，已经创建新的模型支持
| `trainers/mixin.py:_patch_skip_first_batches` | resume 时跳过已训 batch。 | ⚠️ dev resume 语义已有，需确认是否等价 | resume 已经有方案，不需要处理
| `seq2seq_trainer.py:_patch_predict_with_generate` | 评测时用 generate 做预测。 | ⚠️ dev 若支持 predict_with_generate 评测 | 不再支持
| `trainers/arguments.py:_patch_liger_kernel` | 启用 Liger kernel。 | ❌ **twinkle 有 `TransformersFusedCEPatch`（fused_lm_ce）** | 已经在 loss 层面支持
| `trainers/utils.py:patch_modelscope_hub_timeout` | ModelScope hub 超时。 | ⚠️ 通用，dev 拉模型时可能需要 |
| `rlhf_trainers/utils.py:patch_stateless_process_group_for_ipv6` | vLLM 权重同步进程组 IPv6 兼容。 | ⚠️ dev-GRPO + IPv6 |不迁移
| `rlhf_trainers/utils.py:patch_lora_merge` / `patch_lora_unmerge` | rollout 前后合并/拆 LoRA 权重。 | ⚠️ dev-GRPO + LoRA rollout | 分批合并 lora 使用，暂时不迁移
| `rlhf_trainers/utils.py:patch_save_last_checkpoint` | 修最后 ckpt 保存。 | ⚠️ dev-RLHF | trl 的 bug
| `rlhf_trainers/utils.py:patch_vllm_moe_model_weight_loader` | vLLM MoE 权重加载。 | ⚠️ dev-GRPO + MoE | 老版本兼容补丁，不迁移
| `rlhf_trainers/utils.py:patch_vllm_load_adapter` | vLLM 动态加载 LoRA adapter。 | ⚠️ dev-GRPO + LoRA | 不需要，twinkle 有对应能力
| `rlhf_trainers/utils.py:patched_get_lora_tokenizer` | vLLM LoRA tokenizer 兼容。 | ⚠️ dev-GRPO + LoRA | 同上
| `rlhf_trainers/ppo_trainer.py:_patch_dataloader` | PPO collate_fn 兼容。 | ⚠️ dev-PPO | 不需要迁移
| `rlhf_trainers/rlhf_mixin.py:_patch_concatenated_forward` | DPO 类拼接 forward。 | ⚠️ dev-DPO | 不需要迁移

---

## 7. Tuner / PEFT — `tuners/`

| Patch | 功能 / 触发场景 | dev 是否需要 |
|---|---|---|
| `peft.py:_patch_param_wrapper` | 修 peft `ParamWrapper.get_param`。 | ⚠️ dev 用 peft 时（dev adapter 已依赖 peft） | 已迁移
| `peft.py:hot_patch_peft_module` | 热替换 peft 模块 forward（keep-device 等）。 | ⚠️ dev + peft 特定场景 | 不迁移
| `tuners/lora.py:unpatch_lora` | 卸载 swift 自定义 LoRA patch。 | ⚠️ 用 swift LoRA（非 peft）时 | 不是 patch
| `tuners/prompt.py:patch_attention_mask` | prompt-tuning 改 attention mask。 | ⚠️ 用 prompt tuning 时 | 不是 patch

---

## 8. 模板 — `template/`

| Patch | 功能 / 触发场景 | dev 是否需要 |
|---|---|---|
| `template/base.py:_patch_flash_attention_forward` | 改 flash-attn forward 传 position_ids（packing/SP）。 | ⚠️ **dev 复用 legacy template，packing/SP 时可能需要** | 不迁移
| `templates/glm.py:_patch_create_causal_mask` | GLM causal mask 构造兼容。 | ⚠️ 用 GLM 时 |

---

## 9. 量化 — `pipelines/export/quant.py`

| Patch | 功能 / 触发场景 | dev 是否需要 |
|---|---|---|
| `_patch_awq_move_embed` | AWQ 量化时移动 embedding。 | ⚠️ **dev 有 quantize recipe，支持 AWQ 导出时需要** | 已经迁移
| `_patch_gptq` / `_patch_gptq_block` | GPTQ 量化流程/分块兼容。 | ⚠️ dev GPTQ 导出时需要 | 已经迁移

---

## 10. Pipeline / 其他

| Patch | 功能 / 触发场景 | dev 是否需要 |
|---|---|---|
| `pipelines/infer/rollout.py:_patch_full_weight_reload_loader` | rollout 全量权重重载。 | ⚠️ dev-GRPO rollout | 暂时不动，存疑，需要实验
| `pipelines/train/tuner.py:_patch_modules_to_save_zero3` | ZeRO-3 下 `modules_to_save` 兼容。 | ⚠️ dev + DeepSpeed ZeRO-3 + 额外可训模块 | 已迁移

---

## 11. Hub / 参数 / 工具 / 数据集

| Patch | 功能 / 触发场景 | dev 是否需要 |
|---|---|---|
| `hub/hub.py:patch_hub` | 统一 MSHub/HFHub 接口（下载/上传）。 | ✅ **通用能力，dev 拉/推模型仍需要** | trainer 相关，不需要迁移
| `utils/hub_utils.py:patch_kernels` | 进程级替换 `get_kernel`（自定义 kernel 源）。 | ⚠️ 用自定义 kernel 时 |
| `utils/utils.py:_patch_args` / `_patch_get_type_hints` | dataclass 参数解析期的类型提示兼容。 | ✅ **若 dev 复用 swift 参数体系则需要**（通用工具） | 不迁移
| `utils/utils.py:patch_getattr` | 给类装 `__getattr__` 代理（防重复 patch）。 | ⚠️ 通用工具，按需 | 不迁移
| `arguments/base_args/base_args.py:_patch_peft` | 参数初始化期 peft 兼容。 | ⚠️ dev 若走 legacy args + peft | 不需要
| `dataset/preprocessor/core.py:_patch_arrow_writer` | 修 HF datasets ArrowWriter（预处理写盘）。 | ⚠️ **dev 复用数据集预处理链路时需要** | 不迁移

---

## 12. NPU / Ascend — `model/npu_patch/`（整目录，~20 个）

> **整体定位**：Ascend NPU 生态兼容层，覆盖：
> - **env**：`torch_npu` getenv、`HCCL_CONNECT_TIMEOUT` 默认值。
> - **fsdp**：NPU 上 FSDP 兼容。
> - **mindspeed**：MindSpeed 的 FLA/GDN、TE-CP、layernorm-linear frozen weight、GDN-CP helpers 等一系列 Megatron-on-NPU 补丁。
> - **vllm_ascend**：LoRA / memory / MoE / attention 的 vLLM-Ascend runtime 补丁 + MoE 专家 LoRA 训练校验。
> - **model / megatron_checkpoint / utils**：NPU 上模型 forward、flash-linear-attention 可用性、ckpt load、patch-map 应用。

