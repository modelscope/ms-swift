# Clipped Importance Sampling Policy Optimization (CISPO)

**版本依赖**：ms-swift>=3.11

Clipped Importance Sampling Policy Optimization (CISPO) 是 [MiniMax-M1](https://arxiv.org/abs/2506.13585) 论文中提出的一种强化学习算法。相比GRPO（Group Relative Policy Optimization）算法，CISPO 对重要性采样权重（importance sampling weights）本身进行裁剪。

## 算法原理
为便于理解，我们基于 GRPO 算法进行对比说明。

GRPO通过裁剪策略比率来限制策略更新幅度，其损失函数为：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}\left[\min\left(r_t(\theta) \cdot \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot \hat{A}_t\right)\right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是重要性采样比。

在处理长推理链条时，这种裁剪方式可能导致以下问题：

**关键 Token 的梯度被抑制**：在复杂推理任务中，某些关键的低概率 token（如 *However, Recheck, Wait, Aha*）对于触发深度思考和推理纠错至关重要。这些 token 在旧策略 $\pi_{\theta_{\text{old}}}$ 中概率较低，当新策略试图提高其概率时，会导致较大的策略比率 $r_t(\theta)$，GRPO 的裁剪机制会将这些 token 丢弃。


### CISPO 的解决方案

CISPO 的核心思想是：裁剪重要性采样权重，保留梯度更新。具体来说，CISPO 的损失函数为：

$$
\mathcal{L}_{\text{CISPO}}(\theta) = -\mathbb{E}\left[\text{detach}\left(\min(r_t(\theta), \epsilon_{\text{high}})\right) \cdot \hat{A}_t \cdot \log \pi_\theta(a_t|s_t)\right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是重要性采样比。

**关键机制**：
- 对重要性采样权重进行裁剪：$\min(r_t(\theta), \epsilon_{\text{high}})$
- **detach 操作**：裁剪后的权重不参与梯度计算，作为常数系数
- 梯度来自 $\log \pi_\theta(a_t|s_t)$ 项，保证所有 token 都有梯度贡献


## 实现细节
CISPO 的伪代码实现如下：

```python
log_ratio = per_token_logps - old_per_token_logps
importance_weights = torch.exp(log_ratio)  # r_t(θ) = π_θ / π_θ_old

clamped_ratios = torch.clamp(importance_weights, max=epsilon_high).detach()

per_token_loss = -clamped_ratios * advantages.unsqueeze(1) * per_token_logps
```

## 参数设置

我们可以基于 `GRPOTrainer`，通过设置以下参数实现 CISPO 训练：

```bash
--loss_type cispo
--epsilon_high 5.0
```

> 相比其他算法, cispo 的 epsilon_high 一般取值较大，minimax论文中未给出具体的参数设置，这里的值参考论文[ScaleRL](https://arxiv.org/pdf/2510.13786)的实验设置

其他训练参数参考 [GRPO参数文档](../../Command-line-parameters.md#grpo参数)
