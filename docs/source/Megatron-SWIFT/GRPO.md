# GRPO

**版本依赖**：ms-swift >= 3.11

如果你是首次使用 GRPO，请先参考 [GRPO文档](../Instruction/GRPO/GetStarted/GRPO.md)。

Megatron GRPO 当前已支持以下功能：

- **训练模式**：全参数训练与 LoRA 微调
- **并行策略**：支持上下文并行（CP）、流水线并行（PP）、张量并行（TP）和专家并行（EP）
- **推理加速**：支持 vLLM 的 colocate 模式和 server 模式
- **模型支持**：兼容 Megatron Swift 中的 LLM 及 MLLM（多模态大模型）
- **算法支持**：涵盖 swift GRPO 的大部分功能

以下参数或功能将在后续版本中逐步支持：

- **Entropy 相关配置**：如 `top_entropy_quantile`、`log_entropy`
- **Reward Model / Reward Model Plugin**
- **多轮 Rollout 调度机制**（`multi_turn_scheduler`）：实现多轮对话策略优化
- **虚拟流水线并行**（VPP）
- **参考模型同步更新**（`sync_ref_model`）
- **Async Generate** (`async_generate`)
- **num_iterations**
- **日志同步 SwanLab**

⚠️ 注意：以下参数在 Megatron GRPO 中不生效：

- **`use_vllm`**：Megatron GRPO 仅使用 vLLM 进行 Rollout 推理。
- **`move_model_batches`**：该参数专用于 DeepSpeed ZeRO-3 优化，在 Megatron 架构下无效。

与 ms-swift GRPO 相同，Megatron GRPO batch size 相关的参数均以 **completion-level** 为单位，即表示模型生成的 completion 数量，而非 prompt 数量。

#### 参数对比

下表对比了 ms-swift 和 Megatron-SWIFT 中批量相关参数的对应关系：

| ms-swift 参数 | Megatron-SWIFT 参数 | 说明 |
|---------------|---------------------|------|
| `per_device_train_batch_size` | `micro_batch_size` | 每张 GPU 的训练批次大小（completion-level） |
| `gradient_accumulation_steps` | - | 梯度累积步数，在 Megatron-SWIFT 中已包含在 `global_batch_size` 的计算中 |
| - | `global_batch_size` | 全局批次大小（completion-level）<br/>**Megatron-SWIFT**: `micro_batch_size × dp_size × gradient_accumulation_steps`<br/>**ms-swift**: `per_device_train_batch_size × world_size × gradient_accumulation_steps` |
| `num_generations` | `num_generations` | 每个 prompt 生成的 completion 数量 |
| `steps_per_generation` | `steps_per_generation` | Rollout 批次大小相对于训练批次大小的倍数<br/>**注意**：在 ms-swift 中需为 `gradient_accumulation_steps` 的整数倍 |
| `generation_batch_size` | `generation_batch_size` | Rollout 阶段的批次大小（completion-level），需为 `global_batch_size` 的整数倍 |

以下公式用于计算 Megatron GRPO 中的批量：

- **数据并行大小**：`dp_size = world_size / (TP × PP × CP)`
- **全局批次大小**：`global_batch_size = micro_batch_size × dp_size × gradient_accumulation_steps`
- **生成批次大小**：`generation_batch_size = global_batch_size × steps_per_generation`
- **Rollout Prompt 数量**：`num_rollout_prompts = generation_batch_size / num_generations`
- **训练 Prompt 数量**：`num_train_prompts = global_batch_size / num_generations`
- **每个 DP group 的训练 Prompt 数量**：`num_prompts_per_dp_group = global_batch_size / num_generations / dp_size`

注意：在 Megatron GRPO 中，每个 DP group 的训练 Prompt 数量须满足 `num_prompts_per_dp_group` 是 `micro_batch_size`的整数倍，以确保训练批次能够正确分配。

更多参数请参考[命令行文档](./Command-line-parameters.md#grpo参数)

训练脚本请参考[Megatron GRPO 脚本](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/grpo)
