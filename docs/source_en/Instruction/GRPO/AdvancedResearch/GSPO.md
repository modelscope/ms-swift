# Group Sequence Policy Optimization

**Version Requirement**: ms-swift>=3.7

[Group Sequence Policy Optimization](https://www.arxiv.org/abs/2507.18071) points out that in GRPO, importance sampling weights are computed at the token level. However, this approach samples only once per token, making it ineffective for proper distribution correction. Instead, it introduces high-variance noise into the training process, which can destabilize gradient estimation and ultimately cause model collapse. Therefore, the paper argues that the unit of optimization should match the unit of reward. Since rewards are typically assigned at the sequence level (i.e., for the entire generated response), it is more reasonable to perform off-policy correction and optimization at the sequence level, rather than at the token level.

In GRPO, the importance sampling ratio is computed at the token level as follows:

$$
w^{\mathrm{GRPO}}_{i,t} = \frac{\pi_\theta (y_{i, t} \mid x, y_{i, <t})}{\pi_{\theta_{\mathrm{old}}} (y_{i, t} \mid x, y_{i, <t})}
$$

In GSPO, the importance sampling ratio is calculated at the sequence level as:

$$
w^{\mathrm{GSPO}}_{i} = \left[ \frac{\pi_\theta (y_i \mid x)}{\pi_{\theta_{\mathrm{old}}} (y_i \mid x)} \right]^{\frac{1}{|y_i|}}
= \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta (y_{i, t} \mid x, y_{i, <t})}{\pi_{\theta_{\mathrm{old}}} (y_{i, t} \mid x, y_{i, <t})} \right)
$$

Based on GRPO training, we can use the parameter `--importance_sampling_level sequence` to apply the GSPO algorithm.

Other hyperparameters in the paper
```bash
    --epsilon 3e-4 # from paper section 5.1
    --epsilon_high 4e-4 # from paper section 5.1
    --steps_per_generation 4 # from paper section 5.1 (each batch of rollout data is partitioned into four minibatches for gradient updates)
    --beta 0 # zero kl regularization https://github.com/volcengine/verl/pull/2775#issuecomment-3131807306
```
