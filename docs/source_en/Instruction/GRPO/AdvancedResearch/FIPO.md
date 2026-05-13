# FIPO: Future-KL Influenced Policy Optimization

[FIPO](https://arxiv.org/abs/2603.19835) is a value-free RL method for eliciting longer and deeper reasoning. It keeps the GRPO/DAPO training scaffold, but changes how token-level policy updates are weighted: instead of applying one sequence-level advantage uniformly to every token, FIPO uses a discounted Future-KL signal to estimate whether the future trajectory after each token is being reinforced or suppressed.

## Core Idea

In GRPO/DAPO, tokens in the same response usually share the same sequence-level advantage:

$$
\hat{A}_{i,t} = \hat{A}_{i}
$$

This is simple and stable, but the credit assignment is coarse. FIPO starts from the signed log-probability shift between the current policy and the old policy:

$$
\Delta \log p_t = \log \pi_\theta(y_t \mid x, y_{<t}) -
\log \pi_{\mathrm{old}}(y_t \mid x, y_{<t})
$$

A positive value means the token probability is being increased by the current update, while a negative value means it is being suppressed. FIPO then accumulates this signal from the current token to the end of the response:

$$
\mathrm{FutureKL}_t =
\sum_{k=t}^{T}\gamma^{k-t} M_k \Delta \log p_k
$$

where $M_k$ is the completion mask and $\gamma = 2^{-1 / \text{decay\_rate}}$. A larger `decay_rate` gives farther future tokens more influence; a smaller value makes the signal more local. FIPO maps the Future-KL value into a bounded influence weight:

$$
f_t = \mathrm{clip}(\exp(\mathrm{FutureKL}_t), 1-\epsilon_f, 1+\epsilon_f)
$$

The original advantage is then replaced by a future-aware advantage:

$$
\tilde{A}_{i,t} = \hat{A}_{i} \cdot f_{i,t}
$$

## Parameters


| Parameter                 | Type    | Default | Description                                                                            |
| ------------------------- | ------- | ------- | -------------------------------------------------------------------------------------- |
| `--loss_type`             | `str`   | `grpo`  | Set to`fipo` to enable FIPO loss                                                       |
| `--fipo_decay_rate`       | `float` | `32.0`  | Half-life parameter for Future-KL; the actual discount is`2 ** (-1 / fipo_decay_rate)` |
| `--fipo_clip_range`       | `float` | `0.2`   | Influence weight clipping range;`0.2` clips to `[0.8, 1.2]`                            |
| `--fipo_clip_high_only`   | `bool`  | `true`  | If`true`, clips the weight to `[1.0, 1.0 + fipo_clip_range]`                           |
| `--fipo_detach_weight`    | `bool`  | `true`  | Whether to stop gradients through the influence weight                                 |
| `--fipo_safety_threshold` | `float` | `4.0`   | Masks policy loss for negative-advantage tokens whose IS ratio exceeds this threshold  |

## Training Example

[swift](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/internal/fipo.sh)
