# REINFORCE Leave-One-Out (RLOO)

**Version requirement**: ms-swift>=3.10

[REINFORCE Leave-One-Out (RLOO)](https://arxiv.org/abs/2402.14740) is a reinforcement learning algorithm based on the classic REINFORCE policy-gradient method. It constructs an unbiased advantage baseline via the Leave-One-Out (LOO) technique.

## Algorithm Overview

For clarity, we explain RLOO by contrasting it with GRPO (Group Relative Policy Optimization).

Both GRPO and RLOO estimate advantages via intra-group comparisons to avoid the high variance of a global baseline. Their core differences are mainly in the following aspects:

### Difference 1: How the Advantage Baseline Is Constructed

**1. GRPO (Group Relative Policy Optimization)**

For each prompt, GRPO generates $G$ response samples and normalizes rewards using the group mean and standard deviation:

$$
\hat{A}_{i} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}
$$

Where:
- $R_i$ is the reward of the $i$-th sample
- $\text{mean}(\{R_j\}_{j=1}^G) = \frac{1}{G}\sum_{j=1}^G R_j$ is the group mean
- $\text{std}(\{R_j\}_{j=1}^G)$ is the group standard deviation

**2. RLOO (REINFORCE Leave-One-Out)**

For each prompt, RLOO generates $K$ response samples and constructs the baseline via Leave-One-Out, i.e., for the $i$-th sample, the baseline is the mean of the other $K-1$ samples:

$$
\hat{A}_{i} = R_i - \frac{1}{K-1}\sum_{j \neq i} R_j
$$

This can be equivalently rewritten as:

$$
\hat{A}_{i} = \frac{K}{K-1} \left(R_i - \bar{R}\right)
$$

where $\bar{R} = \frac{1}{K}\sum_{j=1}^K R_j$ is the group mean reward.

> Note: We use $K$ here to match the notation in the paper. It has the same meaning as $G$ in GRPO and corresponds to the configuration parameter `num_generations`.

**Why Leave-One-Out?**

The key advantage is unbiasedness. For the $i$-th sample, its reward $R_i$ is independent of the baseline $\frac{1}{K-1}\sum_{j \neq i} R_j$, hence the advantage estimate is unbiased. In contrast, using the mean including itself as the baseline introduces bias.

### Difference 2: How KL Regularization Is Applied

To prevent the policy from drifting too far from the reference policy, both algorithms introduce KL divergence regularization, but in different ways:

**GRPO**: Adds KL divergence as an independent regularization term to the [loss](../GetStarted/GRPO.md#algorithm-overview):

$$
\mathcal{L}(\theta) = -\mathbb{E}\left[\hat{A}_i \log \pi_\theta(a_i|s_i)\right] + \beta \cdot \text{KL}(\pi_\theta \Vert \pi_{\text{ref}})
$$

**RLOO**: Integrates KL divergence directly into the reward, constructing a modified reward:

$$
R'_i = R_i - \beta \cdot \text{KL}(\pi_\theta \Vert \pi_{\text{ref}})
$$

where $\beta$ is the KL coefficient (parameter `beta`), and $\pi_{\text{ref}}$ is the reference policy (typically an SFT model or the initial policy).

## Parameter Configuration

RLOO training can be enabled based on `GRPOTrainer` by setting the following parameters:

```bash
# Basic RLOO configuration
--advantage_estimator rloo  # Use RLOO's leave-one-out advantage estimator
--kl_in_reward true         # Integrate KL divergence into the reward (default for RLOO)
```

You can refer to this [script](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/internal/rloo.sh) for training.

### Important Parameters

- **`--advantage_estimator`**: Choose the advantage estimator
  - `grpo` (default): standardize using group mean and standard deviation
  - `rloo`: construct the baseline via Leave-One-Out

- **`--kl_in_reward`**: Controls where the KL term is applied
  - `false`: KL as a separate regularization term in the loss (GRPO style)
  - `true`: subtract KL directly from the reward to form a modified reward (RLOO style)

- **`--num_generations`**: Number of samples per prompt, i.e., $K$

- **`--beta`**: KL regularization coefficient $\beta$
  - Controls how conservatively the policy updates

Other parameters are consistent with the [GRPO arguments](../../Command-line-parameters.md#grpo-arguments).
