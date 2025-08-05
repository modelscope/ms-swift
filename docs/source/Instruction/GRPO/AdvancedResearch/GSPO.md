# Group Sequence Policy Optimization

**版本依赖**：ms-swift>=3.7

[Group Sequence Policy Optimization](https://www.arxiv.org/abs/2507.18071)中指出GRPO在计算重要性采样权重时，是在token级别进行操作的。然而，这种做法由于每个token仅采样一次，无法实现有效的分布校正，反而会在模型训练过程中引入高方差噪声，极易导致模型的梯度估计不稳定，最终造成模型训练的崩塌。因此，论文认为，优化目标的单位应该与奖励的单位保持一致。由于奖励通常是在序列级别（即完整生成的回复）给出的，因此更合理的做法是将 off-policy 校正和优化也提升到序列级别，而非 token 级别。

GRPO 中，重要性采样比在 token 级别上计算，具体公式为

$$
w^{\mathrm{GRPO}}_{i,t} = \frac{\pi_\theta (y_{i, t} \mid x, y_{i, <t})}{\pi_{\theta_{\mathrm{old}}} (y_{i, t} \mid x, y_{i, <t})}
$$

GSPO 中，重要性采样比在序列级别上计算，具体公式为

$$
w^{\mathrm{GSPO}}_{i} = \left[ \frac{\pi_\theta (y_i \mid x)}{\pi_{\theta_{\mathrm{old}}} (y_i \mid x)} \right]^{\frac{1}{|y_i|}}
= \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta (y_{i, t} \mid x, y_{i, <t})}{\pi_{\theta_{\mathrm{old}}} (y_{i, t} \mid x, y_{i, <t})} \right)
$$

我们可以在 GRPO 训练的基础上，使用参数`--importance_sampling_level sequence` 来使用 GSPO 算法

论文其他超参
```bash
    --epsilon 3e-4 # from paper section 5.1
    --epsilon_high 4e-4 # from paper section 5.1
    --steps_per_generation 4 # from paper section 5.1 (each batch of rollout data is partitioned into four minibatches for gradient updates)
    --beta 0 # zero kl regularization https://github.com/volcengine/verl/pull/2775#issuecomment-3131807306
```
