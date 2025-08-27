# DAPO

[Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)](https://arxiv.org/abs/2503.14476) introduces several tricks based on GRPO, including:
- [Clip Higher](#clip-higher)
- [Dynamic Sampling](#dynamic-sampling)
- [Token level Loss](#token-level-loss)
- [Overlong Filtering](#overlong-filtering)
- [Soft Overlong Punishment](#soft-overlong-punishment)

## Clip Higher
PPO and GRPO use symmetric clipping ranges (e.g., ±0.2) to limit the magnitude of policy updates. While this ensures stability, it also restricts the model's exploratory capabilities. Specifically, when certain tokens have extremely low probabilities under the old policy, even if the current gradient indicates they should be reinforced (A > 0), the maximum increase is strictly limited.

DAPO employs an asymmetric clipping range, raising the upper clipping limit to encourage exploration:
- The upper bound (encouragement side) is relaxed to 0.28.
- The lower bound (suppression side) remains unchanged at 0.2.

In GRPO, the default symmetric clipping range is set using `epsilon`.

Parameters:
- `epsilon_high` sets the upper clipping range, while `epsilon` serves as the lower clipping range.

## Dynamic Sampling
GRPO samples multiple responses per question to compute inter-group advantages:

$$
\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}
$$

However, when all generated outputs {o_i} receive the same reward, the inter-group advantage becomes zero, leading to vanishing gradients and reduced training efficiency.

DAPO addresses this issue with a dynamic sampling strategy:
- Skips data with zero inter-group reward standard deviation during sampling.
- Continues generating samples until the batch is filled.

Parameters:
- `dynamic_sample true` enables dynamic sampling.
- `max_resample_times` sets the maximum number of resampling attempts.

## Token level Loss
GRPO normalizes losses at the sentence level, which introduces bias based on response length.

DAPO uses token-level normalization to avoid this bias in loss calculation.

Parameters:
- `loss_type bnpo` enables token-level normalization.

## Overlong Filtering
DAPO argues that forcibly truncated responses contain high reward noise, making it difficult for the model to distinguish between quality issues and length issues. To address this, DAPO filters out truncated data during training, excluding it from loss computation.

Parameters:
- `overlong_filter` enables filtering of overly long samples.

## Soft Overlong Punishment
Language models often struggle with controlling output length:
- Overly long outputs may be truncated, leading to incorrect judgments of valid content.
- Unconstrained length generation affects practicality and computational efficiency.

DAPO designs a three-stage length penalty function:

$$
R_{\text{length}}(L) =
\begin{cases}
0, & L \leq L_{\text{max}} - L_{\text{cache}} \\[10pt]
\dfrac{(L_{\text{max}} - L_{\text{cache}}) - L}{L_{\text{cache}}}, & L_{\text{max}} - L_{\text{cache}} < L \leq L_{\text{max}} \\[10pt]
-1, &  L > L_{\text{max}}
\end{cases}
$$

When the length falls within the interval $(L_{\text{max}} - L_{\text{cache}} < L \leq L_{\text{max}})$, a linearly increasing penalty is applied. For lengths $(L > L_{\text{max}})$, the maximum penalty (-1) is imposed.

Parameters:
- `reward_funcs soft_overlong` enables this reward function.
- `soft_max_length` sets L_max, which defaults to the model's maximum output length (`max_completion_length`).
- `soft_cache_length` sets L_cache.

## Parameter Settings
In summary, the following parameters can be set based on GRPOTrainer to implement DAPO training.

| Parameter             | Type      | Value       |
|-----------------------|-----------|-------------|
| `--loss_type`         | `str`     | `bnpo`      |
| `--epsilon_high`      | `float`   | `0.28`      |
| `--dynamic_sample`    | `bool`    | `true`      |
| `--max_resample_times`| `int`     | `3`         |
| `--overlong_filter`   | `bool`    | `true`      |
| `--reward_funcs`      | `str`     | `soft_overlong`|
| `--soft_cache_length` | `int`     | `4096`      |
