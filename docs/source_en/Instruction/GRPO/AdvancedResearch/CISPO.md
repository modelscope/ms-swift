# Clipped Importance Sampling Policy Optimization (CISPO)

**Version requirement**: ms-swift>=3.11

Clipped Importance Sampling Policy Optimization (CISPO) is a reinforcement learning algorithm proposed in the [MiniMax-M1](https://arxiv.org/abs/2506.13585) paper. Compared to GRPO (Group Relative Policy Optimization), CISPO clips the importance sampling weights themselves.

## Algorithm Overview

For clarity, we explain CISPO by contrasting it with GRPO.

GRPO limits the magnitude of policy updates by clipping the policy ratio. Its loss function is:

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}\left[\min\left(r_t(\theta) \cdot \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot \hat{A}_t\right)\right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the importance sampling ratio.

When handling long reasoning chains, this clipping approach can lead to the following issues:

**Gradient Suppression of Critical Tokens**: In complex reasoning tasks, certain critical low-probability tokens (such as *However, Recheck, Wait, Aha*) are crucial for triggering deep thinking and reasoning error correction. These tokens have low probability in the old policy $\pi_{\theta_{\text{old}}}$. When the new policy attempts to increase their probability, it results in a large policy ratio $r_t(\theta)$, and GRPO's clipping mechanism will discard these tokens.


### CISPO's Solution

The core idea of CISPO is to clip the importance sampling weights while preserving gradient updates. Specifically, CISPO's loss function is:

$$
\mathcal{L}_{\text{CISPO}}(\theta) = -\mathbb{E}\left[\text{detach}\left(\min(r_t(\theta), \epsilon_{\text{high}})\right) \cdot \hat{A}_t \cdot \log \pi_\theta(a_t|s_t)\right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the importance sampling ratio.

**Key Mechanisms**:
- Clip the importance sampling weights: $\min(r_t(\theta), \epsilon_{\text{high}})$
- **Detach operation**: The clipped weights do not participate in gradient computation and serve as constant coefficients
- Gradients come from the $\log \pi_\theta(a_t|s_t)$ term, ensuring all tokens contribute gradients


## Implementation Details

The pseudo-code implementation of CISPO is as follows:

```python
log_ratio = per_token_logps - old_per_token_logps
importance_weights = torch.exp(log_ratio)  # r_t(θ) = π_θ / π_θ_old

clamped_ratios = torch.clamp(importance_weights, max=epsilon_high).detach()

per_token_loss = -clamped_ratios * advantages.unsqueeze(1) * per_token_logps
```

## Parameter Configuration

CISPO training can be enabled based on `GRPOTrainer` by setting the following parameters:

```bash
--loss_type cispo
--epsilon_high 5.0
```

> Compared to other algorithms, cispo generally uses a larger value for epsilon_high. The minimax paper does not provide specific parameter settings; the value used here refers to the experimental setup in the paper [ScaleRL](https://arxiv.org/pdf/2510.13786).

For other training parameters, refer to the [GRPO parameter documentation](../../Command-line-parameters.md#grpo-arguments).
