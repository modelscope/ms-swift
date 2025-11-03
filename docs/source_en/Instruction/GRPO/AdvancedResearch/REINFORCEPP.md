# REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models

**Version requirement**: ms-swift>=3.10

[REINFORCE++ Baseline](https://arxiv.org/abs/2501.03262) is a simplified version of the REINFORCE++ algorithm, designed for outcome rewards (response-level scalar rewards). Similar to GRPO, it samples multiple model outputs for each prompt and uses an intra-group baseline to estimate advantages. The key difference lies in the statistics used for normalization.

## Algorithm Overview
For clarity, we explain REINFORCE++ Baseline by contrasting it with GRPO (Group Relative Policy Optimization).

Both GRPO and REINFORCE++ Baseline estimate advantages via intra-group comparisons. Their main differences are:

### Difference 1: Statistics Used for Normalization

**GRPO (Group Relative Policy Optimization)**

For each prompt, GRPO generates $G$ response samples and normalizes using the **mean and standard deviation of all samples within the group**:

$$
\hat{A}_{i} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}
$$

When `scale_rewards='batch'` is set, it uses the **batch-level std of original rewards**:

$$
\hat{A}_{i} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^{N})}
$$

where $N$ is the total number of samples in the batch.

**REINFORCE++ Baseline**

For each prompt, REINFORCE++ generates $G$ response samples, subtracts the group mean, and then normalizes using the **standard deviation of the group-mean-subtracted rewards**:

$$
\begin{align}
\tilde{A}_{i} &= R_i - \text{mean}(\{R_j\}_{j=1}^G) \\
\hat{A}_{i} &= \frac{\tilde{A}_{i}}{\text{std}(\{\tilde{A}_k\}_{k=1}^{N})}
\end{align}
$$

where $N$ is the total number of samples in the batch.

**Key Difference**:
- **GRPO**: Uses the std of **original rewards $R$** for normalization
- **REINFORCE++**: Uses the std of **group-mean-subtracted rewards $\tilde{A}$** for normalization

### Difference 2: KL Divergence Regularization

Similar to RLOO, REINFORCE++ Baseline integrates KL divergence directly into the reward:

$$
R'_i = R_i - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})
$$

where $\beta$ is the KL divergence weight coefficient (corresponding to the parameter `beta`), and $\pi_{\text{ref}}$ is the reference policy.

## Parameter Configuration

We can implement REINFORCE++ Baseline training by configuring the following parameters with `GRPOTrainer`:

```bash
--advantage_estimator reinforce_plus_plus
--scale_rewards batch
--kl_in_reward true
```

For training examples, please refer to this [script](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/internal/reinforce_plus_plus.sh)

### Key Parameter Descriptions

- **`--advantage_estimator`**: Selects the advantage estimation method
  - `grpo` (default): Uses the std of original rewards for normalization
  - `reinforce_plus_plus`: Uses the std of group-mean-subtracted rewards for normalization

- **`--kl_in_reward`**: Controls where the KL divergence regularization term is applied
  - `false`: KL divergence is an independent regularization term in the loss function (GRPO default)
  - `true`: KL divergence is subtracted directly from the reward (REINFORCE++ original implementation)

- **`--scale_rewards`**: Controls the normalization method
  - `group` (default): Intra-group normalization
  - `batch`: Global batch-level normalization (REINFORCE++ original implementation)
  - `none`: No normalization

- **`--num_generations`**: Number of samples generated per prompt ($G$)

- **`--beta`**: KL divergence regularization coefficient ($\beta$)

For other parameters, please refer to [GRPO Parameters](../../Command-line-parameters.md#grpo-arguments)
