# Group Sequence Policy Optimization

**版本依赖**：ms-swift>=3.7

[Group Sequence Policy Optimization](https://www.arxiv.org/abs/2507.18071)中指出GRPO在计算重要性采样权重时，是在token级别进行操作的。然而，这种做法由于每个token仅采样一次，无法实现有效的分布校正，反而会在模型训练过程中引入高方差噪声，极易导致模型的梯度估计不稳定，最终造成模型训练的崩塌。因此，论文认为，优化目标的单位应该与奖励的单位保持一致。由于奖励通常是在序列级别（即完整生成的回复）给出的，因此更合理的做法是将 off-policy 校正和优化也提升到序列级别，而非 token 级别。以下是三种计算策略对比：

1. GRPO
对每个 token 独立计算重要性采样比，具体公式为

$$
w^{\mathrm{GRPO}}_{i,t} = \frac{\pi_\theta (y_{i, t} \mid x, y_{i, <t})}{\pi_{\theta_{\mathrm{old}}} (y_{i, t} \mid x, y_{i, <t})}
$$

2. GSPO (Sequence-Level)

在序列级别上计算重要性采样比，具体公式为

$$
w^{\mathrm{GSPO}}_{i} = \left[ \frac{\pi_\theta (y_i \mid x)}{\pi_{\theta_{\mathrm{old}}} (y_i \mid x)} \right]^{\frac{1}{|y_i|}}
= \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta (y_{i, t} \mid x, y_{i, <t})}{\pi_{\theta_{\mathrm{old}}} (y_{i, t} \mid x, y_{i, <t})} \right)
$$

3. GSPO-token
GSPO-token 结合了序列级与 token 级的重要性采样思想

$$
w_{i, t}^{\mathrm{GSPO-token}} = \mathrm{sg}\left[w_i^{\mathrm{GSPO}}\right] \cdot \frac{\pi_{\theta}(y_{i, t} \mid x, y_{i, < t})}{\mathrm{sg}\left[\pi_{\theta}(y_{i, t} \mid x, y_{i, < t})\right]}
$$

其中，$(\mathrm{sg}[\cdot])$ 表示梯度截断（detach()）。

> 注意：根据梯度推导（即论文中的公式(11)和(18)），当各 token 的 advantage 相同时，GSPO-token 与 GSPO 等价。当前的 GRPO 实现中，所有 token 的 advantage 实际上都是基于句子级 reward 并在 group 内进行归一化，因此在这种设置下，GSPO-token 和 GSPO 在理论上是等价的。不过，GSPO-token 为未来更细粒度（token 级别）的 advantage 提供了支持。

伪代码实现
```python
log_ratio = per_token_logps - old_per_token_logps
# GRPO
log_importance_weights = log_ratio

# GSPO (Sequence-Level)
seq_weight = (log_ratio * mask).sum(-1) / mask.sum(-1)
log_importance_weights = seq_weight.unsqueeze(-1)  # (B,1)

# GSPO-token
seq_weight = (log_ratio * mask).sum(-1) / mask.sum(-1)
log_importance_weights = seq_weight.detach().unsqueeze(-1) + (per_token_logps - per_token_logps.detach())

importance_weights = torch.exp(log_importance_weights)
```

我们可以在 GRPO 训练的基础上，通过参数 `--importance_sampling_level` 选择不同的算法：

- `importance_sampling_level token` （默认，GRPO 实现）
- `importance_sampling_level sequence` （GSPO）
- `importance_sampling_level sequence_token` （GSPO-token）

其中 sequence_token 要求 ms-swift > 3.7 （源码安装）

论文其他超参
```bash
    --epsilon 3e-4 # from paper section 5.1
    --epsilon_high 4e-4 # from paper section 5.1
    --steps_per_generation 4 # from paper section 5.1 (each batch of rollout data is partitioned into four minibatches for gradient updates)
    --beta 0 # zero kl regularization https://github.com/volcengine/verl/pull/2775#issuecomment-3131807306
```

训练可以参考该[脚本](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/internal/gspo.sh)
