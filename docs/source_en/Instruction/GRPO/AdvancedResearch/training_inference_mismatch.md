# Training-Inference-Mismatch

**Version Requirement**: ms-swift>=3.11

**TL;DR**: While GRPO introduces vLLM to accelerate the sampling process, it also introduces Training-Inference Mismatch issues that may affect training stability. This document explains the background, causes, and solutions to this problem.

## Background

### Basic Assumptions of GRPO

The training objective of GRPO (Group Relative Policy Optimization) can be expressed as:

$$
\mathcal{L}_{\text{GRPO}} = - \mathbb{E}_{y \sim \pi_\theta} \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t|x, y_{<t})}$ is the importance sampling ratio
- $\hat{A}_t$ is the advantage function, calculated based on reward and group baseline
- $\epsilon$ is the clipping parameter

**Core Assumption**: Samples $y$ are drawn from the policy $\pi_\theta$. In practice, this means:
1. The rollout model and the training model (policy model) should be **the same model** $\pi_\theta$
2. The probability distributions of both models should be **exactly identical**, i.e., $\pi_{\text{rollout}} = \pi_\theta$

### Deviation After Introducing vLLM

GRPO's training speed is largely constrained by the sampling process (rollout). To accelerate this, training frameworks introduce high-performance inference engines (such as vLLM) for sampling. The **ideal assumption** is that through weight synchronization, vLLM maintains consistency with the training model, i.e., $\pi_{\text{vLLM}} \equiv \pi_\theta$.

However, in practice, even with fully synchronized weights, due to differences in operator implementations, the probability distributions still deviate:

$$
\pi_{\text{vLLM}}(y|x) \neq \pi_\theta(y|x)
$$

At this point, the actual training objective becomes:

$$
\mathcal{L} = - \mathbb{E}_{y \sim \pi_{\text{vLLM}}} \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

Where samples come from $\pi_{\text{vLLM}}$, but gradients are computed based on $\pi_\theta$. This **violates the algorithm's on-policy assumption**, introducing training-inference mismatch issues and potentially causing performance degradation.

## Solution

To address training-inference mismatch, we can introduce **Importance Sampling (IS)** correction mechanisms.

### Importance Sampling Correction

The basic idea of importance sampling is: when samples come from distribution $q$ rather than target distribution $p$, we can correct the expectation calculation by introducing weights:

$$
\mathbb{E}_{x \sim p} [f(x)] = \mathbb{E}_{x \sim q} \left[ \frac{p(x)}{q(x)} \cdot f(x) \right]
$$

Applied to the GRPO scenario, the corrected loss function is:

$$
\mathcal{L}_{\text{corrected}} = - \mathbb{E}_{y \sim \pi_{\text{vLLM}}} \left[ w(x, y) \cdot \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

Where $w(x, y)$ is the importance sampling weight used to correct the distribution bias between vLLM and the training model.

Importance sampling weights can be computed and applied at different granularities:

1. **Token-Level**

Compute the importance sampling ratio at each token:

$$
w_{i,t}^{\text{token}} = \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\text{vLLM}}(y_{i,t}|x, y_{i,<t})}
$$

2. **Sequence-Level**

Compute the sequence-level importance sampling ratio, then broadcast to each token:

$$
w_i^{\text{seq}} = \left[ \frac{\pi_\theta(y_i|x)}{\pi_{\text{vLLM}}(y_i|x)} \right]^{\frac{1}{|y_i|}} = \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\text{vLLM}}(y_{i,t}|x, y_{i,<t})} \right)
$$

### Stability Control: Truncate vs. Mask

Excessively large importance sampling weights can cause gradient explosion and destabilize training. Therefore, weight control is necessary:

#### 1. Truncate

Truncate the importance sampling weight to the $[0, \tau]$ interval:

$$
w_{\text{truncate}} = \min(w, \tau)
$$

This method retains all samples but limits their influence.

#### 2. Mask

Discard token/sequence data where weights exceed the threshold:

$$
w_{\text{mask}} = \begin{cases}
w & \text{if } w \leq \tau \\
0 & \text{otherwise}
\end{cases}
$$


### Four Correction Modes

Combining granularity and control strategies, there are four correction modes (selected via `--rollout_importance_sampling_mode` parameter):

| Mode | Description |
|------|-------------|
| `token_truncate` | Token-level truncation |
| `token_mask` | Token-level masking |
| `sequence_truncate` | Sequence-level truncation |
| `sequence_mask` | Sequence-level masking |

The threshold is set via the `--rollout_importance_sampling_threshold` parameter.

## Metrics

To monitor the degree of training-inference mismatch during training, we add the following metrics to the logs (prefixed with `rollout_correction/`):

### 1. KL Divergence

KL divergence measures how much the training policy deviates from the rollout policy. Both metrics estimate $\text{KL}(\pi_\theta \| \pi_{\text{vLLM}})$, which is directly related to the importance sampling ratio $\rho = \frac{\pi_\theta}{\pi_{\text{vLLM}}}$.

**Direct estimator `kl`**:

$$
\text{KL}(\pi_\theta \| \pi_{\text{vLLM}}) = \mathbb{E}_{\pi_{\text{vLLM}}}\left[ \log \frac{\pi_\theta}{\pi_{\text{vLLM}}} \right]
$$

**K3 estimator `k3_kl`**:

$$
\text{KL}(\pi_\theta \| \pi_{\text{vLLM}}) \approx \mathbb{E}_{\pi_{\text{vLLM}}}\left[ \rho - \log \rho - 1 \right], \quad \rho = \frac{\pi_\theta}{\pi_{\text{vLLM}}}
$$

The K3 estimator is more numerically stable when KL values are small and is always non-negative.

### 2. Perplexity (PPL)

Perplexity measures the model's prediction uncertainty for a sequence:

$$
\text{PPL} = \exp\left( -\frac{1}{|y|} \sum_{t=1}^{|y|} \log p(y_t) \right)
$$

Related metrics:
- `training_ppl` / `training_log_ppl`: Training policy PPL and its logarithm
- `rollout_ppl` / `rollout_log_ppl`: Rollout policy PPL and its logarithm
- `log_ppl_diff`: Log PPL difference, positive value means training policy assigns lower probability
- `log_ppl_abs_diff`: Absolute log PPL difference
- `log_ppl_diff_max` / `log_ppl_diff_min`: Max/min of log PPL difference
- `ppl_ratio`: PPL ratio $\frac{\text{PPL}_{\text{training}}}{\text{PPL}_{\text{rollout}}}$

### 3. χ² Divergence (Chi-squared Divergence)

χ² divergence measures the variance of importance sampling weights:

$$
\chi^2(\pi_\theta \| \pi_{\text{vLLM}}) = \mathbb{E}_{\pi_{\text{vLLM}}}\left[ \rho^2 \right] - 1, \quad \rho = \frac{\pi_\theta}{\pi_{\text{vLLM}}}
$$

- `chi2_token`: Token-level χ² divergence, $\mathbb{E}[\rho_t^2] - 1$
- `chi2_seq`: Sequence-level χ² divergence (geometric mean based), $\mathbb{E}[\rho_{\text{geo}}^2] - 1$, where $\rho_{\text{geo}} = \exp(\frac{1}{T}\sum_t \log \rho_t)$

Higher χ² divergence indicates larger IS weight variance and less stable training. `chi2_seq` uses geometric mean instead of product, making it comparable in scale to `chi2_token`.

### 4. Effective Sample Size (ESS)

Effective sample size measures the number of samples that actually contribute after importance sampling:

$$
\text{ESS} = \frac{1}{\mathbb{E}\left[\left(\frac{w}{\mathbb{E}[w]}\right)^2\right]}
$$

A larger ESS value (closer to 1) indicates more uniform importance sampling weight distribution and higher sample utilization efficiency. When all weights are equal (on-policy), ESS = 1; when weights differ significantly (severely off-policy), ESS becomes small.

### 5. IS Weight Statistics

- `is_weight_mean`: Average importance sampling weight, ideal value is 1.0
- `clipped_frac`: Fraction of samples that were truncated or masked


## Usage

### Logging Diagnostic Metrics Only

If you only want to monitor the degree of training-inference mismatch without enabling importance sampling correction, you can set:

```
--log_rollout_offpolicy_metrics true
```

This will log all diagnostic metrics (KL, PPL, χ², etc.) without modifying the loss function.

### Enabling Importance Sampling Correction

Enable the correction mechanism with the following parameters:

```
--rollout_importance_sampling_mode (default None)
--rollout_importance_sampling_threshold (default 2)
```

When `rollout_importance_sampling_mode` is set, diagnostic metrics are automatically logged without needing to set `log_rollout_offpolicy_metrics`.

## References

1. https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda
2. https://fengyao.notion.site/off-policy-rl
3. https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/rollout_corr_helper.py
