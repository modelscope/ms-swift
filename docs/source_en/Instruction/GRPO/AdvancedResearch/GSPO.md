# Group Sequence Policy Optimization

**Version Requirement**: ms-swift>=3.7

In [Group Sequence Policy Optimization](https://www.arxiv.org/abs/2507.18071), it is pointed out that GRPO computes importance sampling weights at the token level. However, this approach is problematic: since each token is only sampled once, it cannot realize effective distribution correction, and instead introduces high-variance noise during training, which can easily lead to unstable gradient estimates and even training collapse. Therefore, the paper argues that the unit of the objective function should be consistent with that of the reward. Since the reward is typically given at the sequence level (i.e., for the entire generated response), it is more reasonable to perform off-policy correction and optimization at the sequence level rather than the token level.

Below are the three main strategies for computing importance sampling weights:

1. GRPO
GRPO computes the importance sampling ratio independently for each token, as follows:

$$
w^{\mathrm{GRPO}}_{i,t} = \frac{\pi_\theta (y_{i, t} \mid x, y_{i, <t})}{\pi_{\theta_{\mathrm{old}}} (y_{i, t} \mid x, y_{i, <t})}
$$

2. GSPO (Sequence-Level)
GSPO calculates the importance sampling ratio at the sequence level, given by:

$$
w^{\mathrm{GSPO}}_{i} = \left[ \frac{\pi_\theta (y_i \mid x)}{\pi_{\theta_{\mathrm{old}}} (y_i \mid x)} \right]^{\frac{1}{|y_i|}}
= \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta (y_{i, t} \mid x, y_{i, <t})}{\pi_{\theta_{\mathrm{old}}} (y_{i, t} \mid x, y_{i, <t})} \right)
$$

3. GSPO-token
GSPO-token combines both sequence-level and token-level importance sampling:

$$
w_{i, t}^{\mathrm{GSPO-token}} = \mathrm{sg}\left[w_i^{\mathrm{GSPO}}\right] \cdot \frac{\pi_{\theta}(y_{i, t} \mid x, y_{i, < t})}{\mathrm{sg}\left[\pi_{\theta}(y_{i, t} \mid x, y_{i, < t})\right]}
$$

where $\mathrm{sg}[\cdot]$ denotes stop-gradient (detach()).

> **NOTE:** According to gradient analysis (i.e., Eqs. (11) and (18) in the paper), when the advantage for each token is identical, GSPO-token is equivalent to GSPO. In the current implementation of GRPO, all token advantages are normalized based on the sentence-level reward within each group. Therefore, in this setting, GSPO-token and GSPO are theoretically equivalent. However, GSPO-token provides support for future fine-grained (token-level) advantages.

Pseudo-code implementation:
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

Based on GRPO training, you can select different algorithms via the `--importance_sampling_level` argument:

- `importance_sampling_level token` (default, GRPO implementation)
- `importance_sampling_level sequence` (GSPO)
- `importance_sampling_level sequence_token` (GSPO-token)


Other hyperparameters in the paper
```bash
    --epsilon 3e-4 # from paper section 5.1
    --epsilon_high 4e-4 # from paper section 5.1
    --steps_per_generation 4 # from paper section 5.1 (each batch of rollout data is partitioned into four minibatches for gradient updates)
    --beta 0 # zero kl regularization https://github.com/volcengine/verl/pull/2775#issuecomment-3131807306
```

For training, you can refer to [this script](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/internal/gspo.sh).
