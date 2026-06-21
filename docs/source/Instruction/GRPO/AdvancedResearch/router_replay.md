# Router Replay (R2/R3)

**TL;DR**: 在 MoE 模型的 RL 训练中，训练引擎与推理引擎之间的路由（expert routing）不一致会显著放大训推偏差，甚至导致训练崩溃。Router Replay 通过在训练前向传播中回放固定的路由掩码来消除这一不一致。根据回放来源的不同，分为 R2（Vanilla Routing Replay）和 R3（Rollout Routing Replay）两种策略。

## Background

### MoE RL 中的三个策略

在 MoE 模型的 GRPO 训练中，存在三个不同阶段的策略，它们共享相同的模型权重，但路由行为可能不同：

| 策略 | 符号 | 路由结果 | 说明 |
|------|------|---------|------|
| **训练策略** | $\pi_\theta$ | $e^{\pi}_t$ | 梯度更新中的模型 |
| **Old Policy** | $\pi_{\theta_{\text{old}}}$ | $e^{\pi}_{\text{old},t}$ | 批次更新前的模型状态 |
| **Rollout Policy** | $\mu_{\theta_{\text{old}}}$ | $e^{\mu}_{\text{old},t}$ | 推理引擎（如vLLM）中的采样策略，权重与 old policy 相同，但 kernel 实现、精度等差异导致路由不同 |

其中，$\pi_{\theta_{\text{old}}}$ 和 $\mu_{\theta_{\text{old}}}$ 的权重在采样时完全一致，但由于推理引擎与训练引擎的实现差异（如算子实现），即使输入相同，路由结果也可能不同。

### 训推不一致的分解

根据 [论文](https://arxiv.org/abs/2507.18071) 的分析，token 级重要性采样比可以分解为两个因子：

$$
\frac{\pi_\theta(y_t|x, y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x, y_{<t})} = \underbrace{\frac{\pi_{\theta_{\text{old}}}(y_t|x, y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x, y_{<t})}}_{\text{training-inference discrepancy}} \times \underbrace{\frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t|x, y_{<t})}}_{\text{policy staleness}}
$$

对于 MoE 模型，专家路由与这两个因子深度耦合：

- **Training-inference discrepancy**：训练引擎和推理引擎的路由不一致（$e^{\pi}_{\text{old},t} \neq e^{\mu}_{\text{old},t}$），放大输出差异
- **Policy staleness**：随着 mini-batch 更新，路由也随之漂移（$e^{\pi}_t \neq e^{\pi}_{\text{old},t}$），进一步偏离采样时的策略

## R2: Vanilla Routing Replay

R2 的核心思想是在梯度更新时，**回放 old policy 在训练引擎中确定的路由**（$e^{\pi}_{\text{old},t}$）。

### 原理

在训练前向传播中，先用 old policy 的权重做一次前向，记录每一层 MoE Router 选择的 expert indices，然后在训练模型 $\pi_\theta$ 的前向传播中强制使用这些 indices：

$$
g_{\text{replay},i} = \frac{I^{\pi}_{\text{old},i} \cdot \exp(s_{\text{train},i})}{\sum_j I^{\pi}_{\text{old},j} \cdot \exp(s_{\text{train},j})}
$$

其中 $I^{\pi}_{\text{old}}$ 是 old policy 的路由掩码，$s_{\text{train}}$ 是训练模型计算的 router logits。softmax 仍作用于训练 logits，因此梯度可以正常回传到 router 权重。

### 特性

| 场景 | 行为 |
|------|------|
| **首个 mini-batch**（on-policy） | $\theta = \theta_{\text{old}}$，故 $e^{\pi}_t = e^{\pi}_{\text{old},t}$，**target policy 不变**（无 bias） |
| **后续 mini-batch**（off-policy） | $\theta \neq \theta_{\text{old}}$，故 $e^{\pi}_t \neq e^{\pi}_{\text{old},t}$，**target policy 被改变**（有 bias），但 policy staleness 被控制 |

## R3: Rollout Routing Replay

R3 的核心思想是在训练前向传播中，**回放 rollout policy 在推理引擎中确定的路由**（$e^{\mu}_{\text{old},t}$）。

### 原理

在推理引擎（如 vLLM）采样时，额外记录每个 token 在每一层 MoE Router 的 expert indices（路由掩码），然后将这些掩码传递到训练引擎，在 $\pi_\theta$ 的前向传播中强制使用：

$$
g_{\text{replay},i} = \frac{I^{\mu}_{\text{old},i} \cdot \exp(s_{\text{train},i})}{\sum_j I^{\mu}_{\text{old},j} \cdot \exp(s_{\text{train},j})}
$$

### 与其他方法的兼容性

- R3 与 **GSPO** 正交，组合使用可进一步提升
- R3 与 **TIS** 组合不一定有增益（R3 已从根源消除不一致，TIS 的额外修正可能多余）
- Router Replay 与 **Clipping** 在 off-policy 训练中缺一不可

## Router Mask Caching

R3 论文还提出路由掩码可以与 KV Cache 一起缓存：对于相同的前缀 token，MoE Router 的输出是确定性的，因此路由掩码可以随 prefix KVCache 一起存储和复用。这在多轮 Agent 场景（tool calling）中尤为重要，避免了重新 prefill 前缀来获取路由掩码，整体 rollout 延迟开销 < 3%。

## Swift 实现

### 参数设置

通过 `--router_replay_mode` 参数选择路由回放策略：

| 参数值 | 说明 |
|--------|------|
| `disabled`（默认） | 不启用路由回放 |
| `R2` | Vanilla Routing Replay：在训练引擎中记录 old policy 路由并回放 |
| `R3` | Rollout Routing Replay：从推理引擎导出路由掩码并在训练中回放 |

环境依赖:

- R3 需要 vLLM ≥ 0.14.0 以支持返回 `routed_experts` 信息。
- 当前 Router Replay 仅在 Megatron backend 下可用，需要 megatron-core ≥ 0.16.0。

### 与训推校正的关系

Router Replay 与 [Training-Inference Mismatch](training_inference_mismatch.md) 中介绍的重要性采样校正（IS correction）是互补的：

- **IS correction**：在 loss 层面通过权重修正训推概率偏差
- **Router Replay**：在模型结构层面通过固定路由消除偏差来源

## 参考资料

1. [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370)
2. [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
3. [Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](https://arxiv.org/abs/2512.01374)
4. [Megatron Core Router Replay Design Document](https://docs.nvidia.com/megatron-core/developer-guide/nightly/api-guide/router_replay.html)
