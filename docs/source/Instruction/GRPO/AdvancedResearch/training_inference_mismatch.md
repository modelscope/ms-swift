# Training-Inference-Mismatch

**版本依赖**：ms-swift>=3.11

**TL;DR**: GRPO 引入 vLLM 加速采样过程的同时，也引入了训练-推理不一致（Training-Inference Mismatch）的问题，从而可能影响训练稳定性。本文将解释这个问题的背景、原因以及相应的解决方案。

## Background

### GRPO 的基本假设

GRPO (Group Relative Policy Optimization) 的训练目标可以表示为：

$$
\mathcal{L}_{\text{GRPO}} = - \mathbb{E}_{y \sim \pi_\theta} \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t|x, y_{<t})}$ 是重要性采样比
- $\hat{A}_t$ 是优势函数（advantage），基于 reward 和 group baseline 计算
- $\epsilon$ 是 clipping 参数

**核心假设**：样本 $y$ 是从策略 $\pi_\theta$ 中采样得到的。在实际训练中，这意味着：
1. 采样模型（rollout model）与训练模型（policy model）应当是**同一个模型** $\pi_\theta$
2. 两个模型的概率分布应当**完全一致**，即 $\pi_{\text{rollout}} = \pi_\theta$

### 引入 vLLM 后的假设偏离

GRPO 的训练速度很大程度上受到采样过程（rollout）的速度制约。为了加速，训练框架引入高效推理引擎（如 vLLM）来执行采样。**理想假设**是：通过权重同步，vLLM 与训练模型保持一致，即 $\pi_{\text{vLLM}} \equiv \pi_\theta$。

然而，在实践中，即使权重完全同步，由于算子实现等差异，两者的概率分布仍然存在偏差：

$$
\pi_{\text{vLLM}}(y|x) \neq \pi_\theta(y|x)
$$

此时，实际的训练目标变为：

$$
\mathcal{L} = - \mathbb{E}_{y \sim \pi_{\text{vLLM}}} \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中样本来自 $\pi_{\text{vLLM}}$，但梯度是基于 $\pi_\theta$ 计算的，这**破坏了算法的 on-policy 假设**，引入了训推不一致的问题。

## Solution

针对训推不一致问题，可以引入**重要性采样（Importance Sampling, IS）**的校正机制。

### 重要性采样校正

重要性采样的基本思想是：当样本来自分布 $q$ 而非目标分布 $p$ 时，可以通过引入权重来修正期望的计算：

$$
\mathbb{E}_{x \sim p} [f(x)] = \mathbb{E}_{x \sim q} \left[ \frac{p(x)}{q(x)} \cdot f(x) \right]
$$

应用到 GRPO 的场景，修正后的损失函数为：

$$
\mathcal{L}_{\text{corrected}} = - \mathbb{E}_{y \sim \pi_{\text{vLLM}}} \left[ w(x, y) \cdot \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中 $w(x, y)$ 是重要性采样权重，用于校正 vLLM 与训练模型之间的分布偏差

重要性采样权重可以在不同粒度上计算和应用：

1. **Token-Level**

每个 token 上计算重要性采样比：

$$
w_{i,t}^{\text{token}} = \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\text{vLLM}}(y_{i,t}|x, y_{i,<t})}
$$

2. **Sequence-Level**

计算序列级别的重要性采样比，然后广播到每个 token：

$$
w_i^{\text{seq}} = \left[ \frac{\pi_\theta(y_i|x)}{\pi_{\text{vLLM}}(y_i|x)} \right]^{\frac{1}{|y_i|}} = \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\text{vLLM}}(y_{i,t}|x, y_{i,<t})} \right)
$$

### 稳定性控制：Truncate vs. Mask

过大的重要性采样权重会导致梯度爆炸，破坏训练稳定性。因此需要对权重进行控制：

#### 1. Truncate（截断）

将重要性采样权重截断到 $[0, \tau]$ 区间：

$$
w_{\text{truncate}} = \min(w, \tau)
$$

该方法保留所有样本，但限制其影响范围。

#### 2. Mask（屏蔽）

舍弃权重超过阈值的 token/sequence 数据

$$
w_{\text{mask}} = \begin{cases}
w & \text{if } w \leq \tau \\
0 & \text{otherwise}
\end{cases}
$$


### 四种校正模式

结合粒度和控制策略，共设置四种校正模式（通过 `--rollout_importance_sampling_mode` 参数选择）：

| 模式 | 说明 |
|------|------|
| `token_truncate` | Token 级截断 |
| `token_mask` | Token 级屏蔽 |
| `sequence_truncate` | Sequence 级截断 |
| `sequence_mask` | Sequence 级屏蔽 |

其中阈值通过 `--rollout_importance_sampling_threshold` 参数设置。

## Metrics

为了监控训练中训推不一致的程度，我们在log中加入以下指标（前缀为 `rollout_correction/`）：

### 1. KL 散度（KL Divergence）

KL 散度衡量训练策略偏离 rollout 策略的程度。两个指标都估计 $\text{KL}(\pi_\theta \| \pi_{\text{vLLM}})$，这与重要性采样权重 $\rho = \frac{\pi_\theta}{\pi_{\text{vLLM}}}$ 直接相关。

**直接估计器 `kl`**：

$$
\text{KL}(\pi_\theta \| \pi_{\text{vLLM}}) = \mathbb{E}_{\pi_{\text{vLLM}}}\left[ \log \frac{\pi_\theta}{\pi_{\text{vLLM}}} \right]
$$

**K3 估计器 `k3_kl`**：

$$
\text{KL}(\pi_\theta \| \pi_{\text{vLLM}}) \approx \mathbb{E}_{\pi_{\text{vLLM}}}\left[ \rho - \log \rho - 1 \right], \quad \rho = \frac{\pi_\theta}{\pi_{\text{vLLM}}}
$$

K3 估计器在 KL 值较小时数值更稳定，且始终非负。

### 2. Perplexity (PPL)

困惑度衡量模型对序列的预测不确定性：

$$
\text{PPL} = \exp\left( -\frac{1}{|y|} \sum_{t=1}^{|y|} \log p(y_t) \right)
$$

相关指标：
- `training_ppl` / `training_log_ppl`：训练策略的 PPL 及其对数
- `rollout_ppl` / `rollout_log_ppl`：rollout 策略的 PPL 及其对数
- `log_ppl_diff`：log PPL 差异，正值表示训练策略分配的概率更低
- `log_ppl_abs_diff`：log PPL 绝对差异
- `log_ppl_diff_max` / `log_ppl_diff_min`：log PPL 差异的最大/最小值
- `ppl_ratio`：PPL 比率 $\frac{\text{PPL}_{\text{training}}}{\text{PPL}_{\text{rollout}}}$

### 3. χ² 散度（Chi-squared Divergence）

χ² 散度衡量重要性采样权重的方差：

$$
\chi^2(\pi_\theta \| \pi_{\text{vLLM}}) = \mathbb{E}_{\pi_{\text{vLLM}}}\left[ \rho^2 \right] - 1, \quad \rho = \frac{\pi_\theta}{\pi_{\text{vLLM}}}
$$

- `chi2_token`：Token 级别 χ² 散度，$\mathbb{E}[\rho_t^2] - 1$
- `chi2_seq`：Sequence 级别 χ² 散度（基于几何平均），$\mathbb{E}[\rho_{\text{geo}}^2] - 1$，其中 $\rho_{\text{geo}} = \exp(\frac{1}{T}\sum_t \log \rho_t)$

χ² 散度越大，表示 IS 权重方差越大，训练越不稳定。`chi2_seq` 使用几何平均而非乘积，使其与 `chi2_token` 在量级上可比较。

### 4. Effective Sample Size (ESS)

有效样本大小衡量重要性采样后实际起作用的样本数量：

$$
\text{ESS} = \frac{1}{\mathbb{E}\left[\left(\frac{w}{\mathbb{E}[w]}\right)^2\right]}
$$

ESS 值越大（接近1），表示重要性采样权重分布越均匀，样本的有效利用率越高。当所有权重相等时（on-policy），ESS = 1；当权重差异很大时（严重 off-policy），ESS 会很小。

### 5. IS 权重统计

- `is_weight_mean`：平均重要性采样权重，理想值为 1.0
- `clipped_frac`：被截断或屏蔽的样本比例


## 使用方式

### 仅记录诊断指标

如果只想监控训推不一致的程度，而不启用重要性采样校正，可以设置：

```
--log_rollout_offpolicy_metrics true
```

这将记录上述所有诊断指标（KL、PPL、χ² 等），但不会对损失函数进行任何修正。

### 启用重要性采样校正

在GRPO训练中，设置以下参数启用校正机制：

```
--rollout_importance_sampling_mode （默认为None）
--rollout_importance_sampling_threshold （默认为2）
```

当设置了 `rollout_importance_sampling_mode` 时，诊断指标会自动记录，无需额外设置 `log_rollout_offpolicy_metrics`。

参考资料

1. https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda
2. https://fengyao.notion.site/off-policy-rl
3. https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/rollout_corr_helper.py
