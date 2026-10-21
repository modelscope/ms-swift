# Swift × Twinkle Template 重构状态

设计以 `TEMPLATE_REDESIGN.md` 为准。本文件只记录当前完成面、验证结果和未完成任务。

## 当前状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| 0：基线与门禁 | 主体完成 | Golden 改为直接对拍 legacy；shift/unshift AST 门禁已接线 |
| 1：数据基础 | 完成 | EncodingOps、row 顺序、12 个 family capability 与通用能力门禁已落地 |
| 2：输出与配置 | 完成 | OutputCodec、TemplateSpec、RolloutState 契约已统一 |
| 3：接口与消费者 | 主体完成 | E1/E3/E7/E9/S4 已退出；HTTP 纯文本多轮保留，state_id 待实现 |
| 4：批处理、媒体、性能 | 未开始 | S1/S2/S3、MediaAdapter、dummy 前移；RenderCache 已降级为不可序列化占位，单渲染优化待定 |
| 5：验收 | 未开始 | 全矩阵数值、轨迹和吞吐验收 |

## 本轮修复

- `DatasetConfig.packing=True` 在验证边界同样推导 `TemplateConfig.padding_free=True`；`process_configs` 仍是首选入口，validate 中的幂等推导待后续收敛到统一 `process_and_validate` 生命周期。
- Swift `InputProcessor` 恢复 Twinkle `_get_packed_seq_params(position_ids)` 的静态签名。
- mixed image/text batch 的缺失 `mm_token_type_ids` 按 `input_ids` 的完整 tensor shape 填充。
- `ClientMultiTurnRollout` 补回输出顺序、stop reason、logprobs/labels、`max_turns=1`、网络异常和完整 trajectory golden；多模态能力模板的纯文本轨迹允许续写。
- 删除 `shift_next_token` / `unshift_next_token` 薄包装；输入转换与 delta render 收敛到 `template_contract/encoding.py`。
- 12 个未接线能力通过 `TemplateCapability` 精确登记：Qwen3-TTS collator/loss、DiffusionGemma loss、5 个 family HF generate、UnlimitedOCR 输出后处理、Qwen PRM decode、Emu3 generate kwargs、Qwen3 reranker engine kwargs；训练/采样入口显式 fail-fast。
- unsupported system、multi-round、task type 和未声明的 multimodal padding-free 组合是额外通用门禁。
- 删除已由 dev processor/engine 覆盖的 family `packing_row`、`forward_context` 和 MiniCPM lmdeploy 重载。

上述三个 packing/多模态失败是代码缺陷，不再归类为 CUDA 环境问题。它们在 GPU 可见环境 3/3 通过；不是纯 CPU 测试。

## 已知基线与环境

- 本轮 Swift processor（强制 processor 使用 CPU、排除 fresh-worker 环境项）：`497 passed`；packing/MM 通用 GPU 逻辑 focused：`35 passed`。
- 本轮 Swift GRPO + infer + deploy fast：`241 passed / 3 failed`，3 项均为既有 infer 日志捕获或 fresh-process CUDA 驱动问题。
- HTTP client focused：`19 passed`；本轮 Twinkle template + client + multi-turn/token-utils：`211 passed / 4 failed`，均为既有 DeepSeek-V4 映射或旧 multi-turn 测试分叉。
- 当前 CUDA 13 构建与主机驱动不兼容仍会阻塞真实 vLLM 0.27 GPU 测试；这不包含上述三个已修复的 CPU/通用 GPU 逻辑测试。
- TransferQueue/tensordict 缺失仍阻塞对应 Twinkle async RL 收集。
- 116/117 family representative 是覆盖门禁，不代表全部 chat-template parity 通过；slow parity 仍有 123 个既有失败。

## 下一步

1. 扩充多模态 family 与 thinking/tool decode 的 legacy oracle。
2. 修复 Qwen3 `last_round` user follow-up 的历史重写能力倒退。
3. 保存 per-sample 渲染选项，使多轮重渲染与首轮 thinking 配置一致。
4. 进入阶段 4，收敛 processor media hooks、batch grouping 和 dummy 注入。

不执行 commit、tag 或 push。
